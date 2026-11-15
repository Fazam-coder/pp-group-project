#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define BLOCK (1 << 20)
#define MAX_SYMBOLS 256
#define MAX_TREE_NODES (MAX_SYMBOLS * 2)

static double now(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

typedef struct {
    int symbol;
    uint32_t freq;
    int left;
    int right;
    int id;
} HNode;

typedef struct {
    uint32_t freq;
    int id;
    int node_idx;
} HeapItem;

typedef struct {
    HeapItem data[MAX_TREE_NODES];
    int size;
} MinHeap;

static inline void heap_push(MinHeap *h, uint32_t freq, int id, int node_idx)
{
    int i = h->size++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].freq < freq || (h->data[p].freq == freq && h->data[p].id <= id))
            break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i].freq = freq;
    h->data[i].id = id;
    h->data[i].node_idx = node_idx;
}

static inline HeapItem heap_pop(MinHeap *h)
{
    HeapItem top = h->data[0];
    HeapItem item = h->data[--h->size];
    if (h->size == 0)
        return top;

    int i = 0;
    while (2 * i + 1 < h->size) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int best = left;
        if (right < h->size) {
            if (h->data[right].freq < h->data[left].freq ||
               (h->data[right].freq == h->data[left].freq && h->data[right].id < h->data[left].id))
                best = right;
        }
        if (item.freq < h->data[best].freq ||
           (item.freq == h->data[best].freq && item.id <= h->data[best].id))
            break;
        h->data[i] = h->data[best];
        i = best;
    }
    h->data[i] = item;
    return top;
}

typedef struct {
    uint8_t symbol;
    uint8_t len;
    uint32_t code;
} CodeInfo;

static int cmp_code_info(const void *a, const void *b)
{
    const CodeInfo *ca = (const CodeInfo *)a;
    const CodeInfo *cb = (const CodeInfo *)b;
    if (ca->len != cb->len)
        return (int)ca->len - (int)cb->len;
    return (int)ca->symbol - (int)cb->symbol;
}

static void compute_lens(const HNode *nodes, int idx, int depth, uint8_t *lens)
{
    if (idx < 0)
        return;
    if (nodes[idx].symbol >= 0) {
        lens[nodes[idx].symbol] = (uint8_t)(depth > 0 ? depth : 1);
        return;
    }
    compute_lens(nodes, nodes[idx].left, depth + 1, lens);
    compute_lens(nodes, nodes[idx].right, depth + 1, lens);
}

static int compress_block(const uint8_t *in, int n, uint8_t *out)
{
    uint32_t orig_size = (uint32_t)n;
    int o = 0;

    memcpy(out + o, &orig_size, sizeof(uint32_t));
    o += sizeof(uint32_t);

    if (n == 0)
        return o;

    uint32_t freq[MAX_SYMBOLS] = {0};
    for (int i = 0; i < n; i++)
        freq[in[i]]++;

    int K = 0;
    int sym_list[MAX_SYMBOLS];
    for (int s = 0; s < MAX_SYMBOLS; s++) {
        if (freq[s] > 0)
            sym_list[K++] = s;
    }

    uint16_t num_sym = (uint16_t)K;
    memcpy(out + o, &num_sym, sizeof(uint16_t));
    o += sizeof(uint16_t);

    CodeInfo code_info[MAX_SYMBOLS];

    if (K == 1) {
        code_info[0].symbol = (uint8_t)sym_list[0];
        code_info[0].len = 1;
        code_info[0].code = 0;

        out[o++] = code_info[0].symbol;
        out[o++] = code_info[0].len;

        int bit_bytes = (n + 7) / 8;
        memset(out + o, 0, bit_bytes);
        o += bit_bytes;
        return o;
    }

    HNode nodes[MAX_TREE_NODES];
    memset(nodes, 0, sizeof(nodes));
    MinHeap heap;
    memset(&heap, 0, sizeof(heap));

    for (int i = 0; i < K; i++) {
        int s = sym_list[i];
        nodes[i].symbol = s;
        nodes[i].freq = freq[s];
        nodes[i].left = -1;
        nodes[i].right = -1;
        nodes[i].id = i;
        heap_push(&heap, freq[s], i, i);
    }

    int next_node = K;
    while (heap.size > 1) {
        HeapItem it1 = heap_pop(&heap);
        HeapItem it2 = heap_pop(&heap);

        int p = next_node++;
        nodes[p].symbol = -1;
        nodes[p].freq = it1.freq + it2.freq;
        nodes[p].left = it1.node_idx;
        nodes[p].right = it2.node_idx;
        nodes[p].id = p;

        heap_push(&heap, nodes[p].freq, p, p);
    }

    uint8_t lens[MAX_SYMBOLS] = {0};
    compute_lens(nodes, heap.data[0].node_idx, 0, lens);

    for (int i = 0; i < K; i++) {
        code_info[i].symbol = (uint8_t)sym_list[i];
        code_info[i].len = lens[sym_list[i]];
    }

    qsort(code_info, K, sizeof(CodeInfo), cmp_code_info);

    uint32_t code = 0;
    int prev_len = 0;
    for (int i = 0; i < K; i++) {
        int cur_len = code_info[i].len;
        if (prev_len > 0)
            code = (code + 1) << (cur_len - prev_len);
        code_info[i].code = code;
        prev_len = cur_len;
    }

    for (int i = 0; i < K; i++) {
        out[o++] = code_info[i].symbol;
        out[o++] = code_info[i].len;
    }

    uint32_t sym_code[MAX_SYMBOLS];
    uint8_t sym_len[MAX_SYMBOLS];
    for (int i = 0; i < K; i++) {
        sym_code[code_info[i].symbol] = code_info[i].code;
        sym_len[code_info[i].symbol] = code_info[i].len;
    }

    uint64_t bit_buf = 0;
    int bit_count = 0;
    for (int i = 0; i < n; i++) {
        uint8_t s = in[i];
        uint32_t c = sym_code[s];
        int l = sym_len[s];

        bit_buf = (bit_buf << l) | c;
        bit_count += l;

        while (bit_count >= 8) {
            bit_count -= 8;
            out[o++] = (uint8_t)(bit_buf >> bit_count);
        }
    }
    if (bit_count > 0) {
        out[o++] = (uint8_t)(bit_buf << (8 - bit_count));
    }

    return o;
}

typedef struct {
    int left;
    int right;
    int symbol;
} DecNode;

static void decompress_block(const uint8_t *in, int csize, uint8_t *out)
{
    if (csize < (int)sizeof(uint32_t))
        return;

    uint32_t orig_size;
    memcpy(&orig_size, in, sizeof(uint32_t));
    if (orig_size == 0)
        return;

    int pos = sizeof(uint32_t);
    uint16_t K;
    memcpy(&K, in + pos, sizeof(uint16_t));
    pos += sizeof(uint16_t);

    CodeInfo code_info[MAX_SYMBOLS];
    for (int i = 0; i < K; i++) {
        code_info[i].symbol = in[pos++];
        code_info[i].len = in[pos++];
    }

    if (K == 1) {
        memset(out, code_info[0].symbol, orig_size);
        return;
    }

    qsort(code_info, K, sizeof(CodeInfo), cmp_code_info);

    uint32_t code = 0;
    int prev_len = 0;
    for (int i = 0; i < K; i++) {
        int cur_len = code_info[i].len;
        if (prev_len > 0)
            code = (code + 1) << (cur_len - prev_len);
        code_info[i].code = code;
        prev_len = cur_len;
    }

    DecNode tree[MAX_TREE_NODES];
    int tree_size = 1;
    tree[0].left = -1;
    tree[0].right = -1;
    tree[0].symbol = -1;

    for (int i = 0; i < K; i++) {
        uint32_t c = code_info[i].code;
        int len = code_info[i].len;
        int sym = code_info[i].symbol;
        int curr = 0;
        for (int b = len - 1; b >= 0; b--) {
            int bit = (int)((c >> b) & 1);
            if (bit == 0) {
                if (tree[curr].left == -1) {
                    int nxt = tree_size++;
                    tree[nxt].left = -1;
                    tree[nxt].right = -1;
                    tree[nxt].symbol = -1;
                    tree[curr].left = nxt;
                }
                curr = tree[curr].left;
            } else {
                if (tree[curr].right == -1) {
                    int nxt = tree_size++;
                    tree[nxt].left = -1;
                    tree[nxt].right = -1;
                    tree[nxt].symbol = -1;
                    tree[curr].right = nxt;
                }
                curr = tree[curr].right;
            }
        }
        tree[curr].symbol = sym;
    }

    int fast_sym[256];
    int fast_len[256];
    for (int i = 0; i < 256; i++) {
        fast_sym[i] = -1;
        fast_len[i] = 0;
    }
    for (int i = 0; i < K; i++) {
        int len = code_info[i].len;
        if (len <= 8) {
            uint32_t c = code_info[i].code;
            int shift = 8 - len;
            uint32_t base = c << shift;
            int count = 1 << shift;
            for (int m = 0; m < count; m++) {
                fast_sym[base + m] = code_info[i].symbol;
                fast_len[base + m] = len;
            }
        }
    }

    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t out_pos = 0;

    while (out_pos < orig_size) {
        while (bit_count <= 48 && pos < csize) {
            bit_buf = (bit_buf << 8) | in[pos++];
            bit_count += 8;
        }

        if (bit_count >= 8) {
            uint8_t byte = (uint8_t)((bit_buf >> (bit_count - 8)) & 0xFF);
            int sym = fast_sym[byte];
            if (sym >= 0) {
                out[out_pos++] = (uint8_t)sym;
                bit_count -= fast_len[byte];
                continue;
            }
        }

        int curr = 0;
        while (tree[curr].symbol < 0) {
            if (bit_count == 0) {
                if (pos < csize) {
                    bit_buf = (bit_buf << 8) | in[pos++];
                    bit_count += 8;
                } else {
                    break;
                }
            }
            bit_count--;
            int bit = (int)((bit_buf >> bit_count) & 1);
            curr = bit ? tree[curr].right : tree[curr].left;
            if (curr < 0)
                break;
        }
        if (curr >= 0 && tree[curr].symbol >= 0) {
            out[out_pos++] = (uint8_t)tree[curr].symbol;
        } else {
            break;
        }
    }
}

static uint8_t *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    *size = (size_t)ftell(f);
    rewind(f);
    uint8_t *buf = malloc(*size + 1);
    if (*size > 0 && fread(buf, 1, *size, f) != *size) {
        fprintf(stderr, "ошибка чтения %s\n", path);
        exit(1);
    }
    fclose(f);
    return buf;
}

static void compress_file(const char *inpath, const char *outpath, int threads)
{
    size_t size;
    uint8_t *in = read_file(inpath, &size);
    size_t nblocks = (size + BLOCK - 1) / BLOCK;
    if (nblocks == 0)
        nblocks = 1;

    uint8_t **res = malloc((nblocks + 1) * sizeof(uint8_t *));
    int *csize = malloc((nblocks + 1) * sizeof(int));

    double t0 = now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)nblocks; b++) {
        size_t off = (size_t)b * BLOCK;
        int n = (int)(off < size ? (size - off < BLOCK ? size - off : BLOCK) : 0);
        res[b] = malloc(2 * (size_t)n + 1024);
        csize[b] = compress_block(in + off, n, res[b]);
    }

    double t1 = now();

    FILE *f = fopen(outpath, "wb");
    if (!f) {
        perror(outpath);
        exit(1);
    }
    fwrite(&size, sizeof(size_t), 1, f);
    fwrite(&nblocks, sizeof(size_t), 1, f);
    fwrite(csize, sizeof(int), nblocks, f);
    size_t total = 2 * sizeof(size_t) + nblocks * sizeof(int);
    for (size_t b = 0; b < nblocks; b++) {
        fwrite(res[b], 1, csize[b], f);
        total += csize[b];
        free(res[b]);
    }
    fclose(f);

    fprintf(stderr, "in=%zu out=%zu ratio=%.3f\n", size, total,
            size ? (double)total / size : 0.0);
    fprintf(stderr, "threads=%d compute=%.4f s\n", threads, t1 - t0);
    free(res);
    free(csize);
    free(in);
}

static void decompress_file(const char *inpath, const char *outpath, int threads)
{
    size_t fsize;
    uint8_t *in = read_file(inpath, &fsize);

    if (fsize < 2 * sizeof(size_t)) {
        fprintf(stderr, "Некорректный формат файла %s\n", inpath);
        free(in);
        exit(1);
    }

    size_t size, nblocks;
    memcpy(&size, in, sizeof(size_t));
    memcpy(&nblocks, in + sizeof(size_t), sizeof(size_t));

    if (fsize < 2 * sizeof(size_t) + nblocks * sizeof(int)) {
        fprintf(stderr, "Поврежденный заголовок файла %s\n", inpath);
        free(in);
        exit(1);
    }

    int *csize = malloc((nblocks + 1) * sizeof(int));
    memcpy(csize, in + 2 * sizeof(size_t), nblocks * sizeof(int));

    size_t *offs = malloc((nblocks + 1) * sizeof(size_t));
    size_t pos = 2 * sizeof(size_t) + nblocks * sizeof(int);
    for (size_t b = 0; b < nblocks; b++) {
        offs[b] = pos;
        pos += csize[b];
    }

    uint8_t *out = malloc(size + 1);

    double t0 = now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)nblocks; b++) {
        decompress_block(in + offs[b], csize[b], out + (size_t)b * BLOCK);
    }

    double t1 = now();

    FILE *f = fopen(outpath, "wb");
    if (!f) {
        perror(outpath);
        exit(1);
    }
    if (size > 0)
        fwrite(out, 1, size, f);
    fclose(f);

    fprintf(stderr, "threads=%d compute=%.4f s\n", threads, t1 - t0);
    free(offs);
    free(csize);
    free(out);
    free(in);
}

static int verify_file(const char *inpath, int threads)
{
    size_t size;
    uint8_t *in = read_file(inpath, &size);
    size_t nblocks = (size + BLOCK - 1) / BLOCK;
    if (nblocks == 0)
        nblocks = 1;

    uint8_t **res = malloc((nblocks + 1) * sizeof(uint8_t *));
    int *csize = malloc((nblocks + 1) * sizeof(int));
    uint8_t *out = malloc(size + 1);

    double t0 = now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)nblocks; b++) {
        size_t off = (size_t)b * BLOCK;
        int n = (int)(off < size ? (size - off < BLOCK ? size - off : BLOCK) : 0);
        res[b] = malloc(2 * (size_t)n + 1024);
        csize[b] = compress_block(in + off, n, res[b]);
    }

    double t1 = now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)nblocks; b++) {
        decompress_block(res[b], csize[b], out + (size_t)b * BLOCK);
    }

    double t2 = now();

    size_t total = 0;
    for (size_t b = 0; b < nblocks; b++) {
        total += csize[b];
        free(res[b]);
    }

    int ok = (size == 0) || (memcmp(in, out, size) == 0);
    printf("%s: %s (in=%zu compressed=%zu, compress=%.4f s, decompress=%.4f s, threads=%d)\n",
           inpath, ok ? "OK" : "ОШИБКА", size, total, t1 - t0, t2 - t1, threads);

    free(res);
    free(csize);
    free(out);
    free(in);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "Использование:\n"
                "  %s c input output [потоки]\n"
                "  %s d input output [потоки]\n"
                "  %s v input [потоки]\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    if (strcmp(argv[1], "v") == 0) {
        if (argc > 3)
            threads = atoi(argv[3]);
        if (threads < 1)
            threads = 1;
        return verify_file(argv[2], threads);
    }

    if (argc < 4) {
        fprintf(stderr, "Нужны input и output\n");
        return 1;
    }
    if (argc > 4)
        threads = atoi(argv[4]);
    if (threads < 1)
        threads = 1;

    if (strcmp(argv[1], "c") == 0)
        compress_file(argv[2], argv[3], threads);
    else
        decompress_file(argv[2], argv[3], threads);
    return 0;
}
