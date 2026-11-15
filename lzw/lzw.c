/* LZW with independent 1 MiB blocks and OpenMP parallelism. */
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define BLOCK (1u << 20)
#define CODE_BITS 12
#define DICT_SIZE (1u << CODE_BITS)
#define HASH_SIZE 8192u
#define MAX_PACKED ((BLOCK * CODE_BITS + 7) / 8)

typedef struct {
    uint8_t *data;
    uint32_t size;
    int ok;
} PackedBlock;

static double now(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static void put32(uint8_t *p, uint32_t x)
{
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(x >> (8 * i));
}

static void put64(uint8_t *p, uint64_t x)
{
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(x >> (8 * i));
}

static uint32_t get32(const uint8_t *p)
{
    uint32_t x = 0;
    for (int i = 0; i < 4; ++i) x |= (uint32_t)p[i] << (8 * i);
    return x;
}

static uint64_t get64(const uint8_t *p)
{
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) x |= (uint64_t)p[i] << (8 * i);
    return x;
}

/* Every dictionary key is (prefix code, next byte). -1 denotes an empty slot. */
static unsigned slot_for(uint32_t key)
{
    return (key * 2654435761u) & (HASH_SIZE - 1);
}

static void emit_code(uint8_t *out, uint32_t *used, uint32_t *bits,
                      int *nbits, uint32_t code)
{
    *bits |= code << *nbits;
    *nbits += CODE_BITS;
    while (*nbits >= 8) {
        out[(*used)++] = (uint8_t)*bits;
        *bits >>= 8;
        *nbits -= 8;
    }
}

static PackedBlock compress_block(const uint8_t *in, uint32_t n)
{
    PackedBlock result = {NULL, 0, 0};
    uint8_t *out = malloc((size_t)MAX_PACKED + 1);
    int32_t *keys = malloc(HASH_SIZE * sizeof(*keys));
    uint16_t *values = malloc(HASH_SIZE * sizeof(*values));
    if (!out || !keys || !values) goto done;
    for (unsigned i = 0; i < HASH_SIZE; ++i) keys[i] = -1;

    uint32_t used = 0, bits = 0, next = 256;
    int nbits = 0;
    if (n) {
        uint32_t prefix = in[0];
        for (uint32_t i = 1; i < n; ++i) {
            uint32_t key = (prefix << 8) | in[i];
            unsigned slot = slot_for(key);
            while (keys[slot] >= 0 && (uint32_t)keys[slot] != key)
                slot = (slot + 1) & (HASH_SIZE - 1);
            if (keys[slot] >= 0) {
                prefix = values[slot];
            } else {
                emit_code(out, &used, &bits, &nbits, prefix);
                if (next < DICT_SIZE) {
                    keys[slot] = (int32_t)key;
                    values[slot] = (uint16_t)next++;
                }
                prefix = in[i];
            }
        }
        emit_code(out, &used, &bits, &nbits, prefix);
    }
    if (nbits) out[used++] = (uint8_t)bits;
    result.data = out;
    result.size = used;
    result.ok = 1;
done:
    if (!result.ok) free(out);
    free(keys);
    free(values);
    return result;
}

static int read_code(const uint8_t *in, uint32_t size, uint32_t *pos,
                     uint32_t *bits, int *nbits, uint32_t *code)
{
    while (*nbits < CODE_BITS && *pos < size) {
        *bits |= (uint32_t)in[(*pos)++] << *nbits;
        *nbits += 8;
    }
    if (*nbits < CODE_BITS) return 0;
    *code = *bits & (DICT_SIZE - 1);
    *bits >>= CODE_BITS;
    *nbits -= CODE_BITS;
    return 1;
}

static int decompress_block(const uint8_t *in, uint32_t size,
                            uint8_t *out, uint32_t expected)
{
    uint16_t prefix[DICT_SIZE];
    uint8_t suffix[DICT_SIZE], stack[DICT_SIZE];
    uint32_t pos = 0, bits = 0, code, next = 256, written = 0, codes = 0;
    int nbits = 0;
    if (!read_code(in, size, &pos, &bits, &nbits, &code))
        return expected == 0 && size == 0;
    if (code >= 256 || expected == 0) return 0;
    ++codes;
    out[written++] = (uint8_t)code;
    uint32_t previous = code;

    while (read_code(in, size, &pos, &bits, &nbits, &code)) {
        ++codes;
        if (code > next || code >= DICT_SIZE || (code == next && next == DICT_SIZE))
            return 0;
        uint32_t phrase = code == next ? previous : code;
        unsigned depth = 0;
        while (phrase >= 256) {
            if (phrase >= next || depth >= DICT_SIZE - 1) return 0;
            stack[depth++] = suffix[phrase];
            phrase = prefix[phrase];
        }
        uint8_t first = (uint8_t)phrase;
        stack[depth++] = first;
        uint32_t length = depth + (code == next);
        if (length > expected - written) return 0;
        while (depth) out[written++] = stack[--depth];
        if (code == next) out[written++] = first;
        if (next < DICT_SIZE) {
            prefix[next] = (uint16_t)previous;
            suffix[next] = first;
            ++next;
        }
        previous = code;
    }
    return written == expected && bits == 0 &&
           size == (codes * CODE_BITS + 7) / 8;
}

static uint8_t *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); fclose(f); return NULL; }
    long length = ftell(f);
    if (length < 0 || fseek(f, 0, SEEK_SET) != 0) {
        perror(path); fclose(f); return NULL;
    }
    *size = (size_t)length;
    uint8_t *data = malloc(*size ? *size : 1);
    if (!data) { fprintf(stderr, "Недостаточно памяти\n"); fclose(f); return NULL; }
    if (fread(data, 1, *size, f) != *size) {
        perror(path); free(data); fclose(f); return NULL;
    }
    fclose(f);
    return data;
}

static int write_file(const char *path, const uint8_t *data, size_t size)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }
    int ok = fwrite(data, 1, size, f) == size;
    if (fclose(f) != 0) ok = 0;
    if (!ok) perror(path);
    return ok;
}

static size_t block_count(size_t size)
{
    return size / BLOCK + (size % BLOCK != 0);
}

static uint32_t block_length(size_t size, size_t b)
{
    size_t remaining = size - b * BLOCK;
    return (uint32_t)(remaining < BLOCK ? remaining : BLOCK);
}

static void free_blocks(PackedBlock *blocks, size_t count)
{
    if (!blocks) return;
    for (size_t b = 0; b < count; ++b) free(blocks[b].data);
    free(blocks);
}

static PackedBlock *compress_blocks(const uint8_t *in, size_t size,
                                    size_t count, int threads)
{
    PackedBlock *blocks = calloc(count ? count : 1, sizeof(*blocks));
    if (!blocks) return NULL;
    double t0 = now();
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)count; ++b)
        blocks[b] = compress_block(in + (size_t)b * BLOCK, block_length(size, b));
    fprintf(stderr, "threads=%d compress=%.4f s\n", threads, now() - t0);
    for (size_t b = 0; b < count; ++b) {
        if (!blocks[b].ok) {
            fprintf(stderr, "Недостаточно памяти при сжатии блока %zu\n", b);
            free_blocks(blocks, count);
            return NULL;
        }
    }
    return blocks;
}

static int compress_file(const char *input, const char *output, int threads)
{
    size_t size;
    uint8_t *in = read_file(input, &size);
    if (!in) return 1;
    size_t count = block_count(size);
    if (count > UINT32_MAX || count > LONG_MAX || count > (SIZE_MAX - 16) / 4) {
        fprintf(stderr, "Слишком большой файл\n"); free(in); return 1;
    }
    PackedBlock *blocks = compress_blocks(in, size, count, threads);
    free(in);
    if (!blocks) return 1;
    size_t total = 16 + 4 * count;
    for (size_t b = 0; b < count; ++b) {
        if (blocks[b].size > SIZE_MAX - total) {
            fprintf(stderr, "Слишком большой файл\n"); free_blocks(blocks, count); return 1;
        }
        total += blocks[b].size;
    }
    uint8_t *packed = malloc(total);
    if (!packed) { fprintf(stderr, "Недостаточно памяти\n"); free_blocks(blocks, count); return 1; }
    memcpy(packed, "LZW1", 4);
    put64(packed + 4, size);
    put32(packed + 12, (uint32_t)count);
    size_t pos = 16 + 4 * count;
    for (size_t b = 0; b < count; ++b) {
        put32(packed + 16 + 4 * b, blocks[b].size);
        memcpy(packed + pos, blocks[b].data, blocks[b].size);
        pos += blocks[b].size;
    }
    int ok = write_file(output, packed, total);
    free(packed);
    free_blocks(blocks, count);
    return ok ? 0 : 1;
}

static int decompress_file(const char *input, const char *output, int threads)
{
    size_t packed_size;
    uint8_t *packed = read_file(input, &packed_size);
    if (!packed) return 1;
    int status = 1;
    if (packed_size < 16 || memcmp(packed, "LZW1", 4) != 0) goto invalid;
    uint64_t original = get64(packed + 4);
    uint32_t count = get32(packed + 12);
    if (original > SIZE_MAX || count != block_count((size_t)original) ||
        count > LONG_MAX || count > (packed_size - 16) / 4) goto invalid;
    size_t *offsets = malloc((count ? count : 1) * sizeof(*offsets));
    int *ok = malloc((count ? count : 1) * sizeof(*ok));
    uint8_t *out = malloc(original ? (size_t)original : 1);
    if (!offsets || !ok || !out) {
        fprintf(stderr, "Недостаточно памяти\n");
        free(offsets); free(ok); free(out); goto done;
    }
    size_t pos = 16 + 4 * (size_t)count;
    for (uint32_t b = 0; b < count; ++b) {
        uint32_t length = get32(packed + 16 + 4 * (size_t)b);
        if (length > MAX_PACKED || length > packed_size - pos) {
            free(offsets); free(ok); free(out); goto invalid;
        }
        offsets[b] = pos;
        pos += length;
    }
    if (pos != packed_size) {
        free(offsets); free(ok); free(out); goto invalid;
    }
    double t0 = now();
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)count; ++b) {
        uint32_t length = get32(packed + 16 + 4 * (size_t)b);
        ok[b] = decompress_block(packed + offsets[b], length,
                                 out + (size_t)b * BLOCK,
                                 block_length((size_t)original, b));
    }
    fprintf(stderr, "threads=%d decompress=%.4f s\n", threads, now() - t0);
    for (uint32_t b = 0; b < count; ++b) {
        if (!ok[b]) {
            fprintf(stderr, "Повреждён блок %u\n", b);
            free(offsets); free(ok); free(out); goto done;
        }
    }
    status = write_file(output, out, (size_t)original) ? 0 : 1;
    free(offsets); free(ok); free(out);
    goto done;
invalid:
    fprintf(stderr, "Некорректный формат файла %s\n", input);
done:
    free(packed);
    return status;
}

static int verify_file(const char *input, int threads)
{
    size_t size;
    uint8_t *in = read_file(input, &size);
    if (!in) return 1;
    size_t count = block_count(size);
    if (count > LONG_MAX) { fprintf(stderr, "Слишком большой файл\n"); free(in); return 1; }
    double t0 = now();
    PackedBlock *blocks = compress_blocks(in, size, count, threads);
    if (!blocks) { free(in); return 1; }
    double t1 = now();
    uint8_t *out = malloc(size ? size : 1);
    int *ok = malloc((count ? count : 1) * sizeof(*ok));
    if (!out || !ok) {
        fprintf(stderr, "Недостаточно памяти\n");
        free(out); free(ok); free_blocks(blocks, count); free(in); return 1;
    }
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)count; ++b)
        ok[b] = decompress_block(blocks[b].data, blocks[b].size,
                                 out + (size_t)b * BLOCK, block_length(size, b));
    double t2 = now();
    size_t compressed = 16 + 4 * count;
    int valid = 1;
    for (size_t b = 0; b < count; ++b) {
        compressed += blocks[b].size;
        if (!ok[b]) valid = 0;
    }
    if (valid && size && memcmp(in, out, size) != 0) valid = 0;
    printf("%s: %s (in=%zu compressed=%zu, compress=%.4f s, decompress=%.4f s, threads=%d)\n",
           input, valid ? "OK" : "ОШИБКА", size, compressed, t1 - t0, t2 - t1, threads);
    free(ok); free(out); free_blocks(blocks, count); free(in);
    return valid ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Использование:\n  %s c input output [потоки]\n"
                "  %s d input output [потоки]\n  %s v input [потоки]\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (strcmp(argv[1], "v") == 0) {
        if (argc > 4) return 1;
        if (argc == 4) threads = atoi(argv[3]);
        if (threads < 1) threads = 1;
        return verify_file(argv[2], threads);
    }
    if (argc < 4) { fprintf(stderr, "Нужны input и output\n"); return 1; }
    if (argc == 5) threads = atoi(argv[4]);
    if (threads < 1) threads = 1;
    if (strcmp(argv[1], "c") == 0) return compress_file(argv[2], argv[3], threads);
    if (strcmp(argv[1], "d") == 0) return decompress_file(argv[2], argv[3], threads);
    fprintf(stderr, "Неизвестная команда %s\n", argv[1]);
    return 1;
}
