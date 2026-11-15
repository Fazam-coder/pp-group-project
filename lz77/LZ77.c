/*
 * LZ77 с хэш-цепочками + OpenMP (параллельность по блокам).
 *
 * Сборка:
 *   параллельная:     gcc -O2 -fopenmp -o lz77_par LZ77.c
 *   последовательная: gcc -O2 -Wno-unknown-pragmas -o lz77_seq LZ77.c
 *
 * Запуск:
 *   lz77_par c input output [потоки]   - сжать
 *   lz77_par d input output [потоки]   - распаковать
 *   lz77_par v input [потоки]          - проверка (сжать, распаковать, сравнить)
 *
 * Формат файла: size_t размер | size_t число блоков | int размеры блоков[] | данные
 * Элементы блока: литерал = 0, байт; match = 1, расстояние (2 байта), длина-3 (1 байт)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define BLOCK (1 << 20)
#define WINDOW 32768
#define MIN_MATCH 3
#define MAX_MATCH 258
#define MAX_CHAIN 64
#define HASH_BITS 15
#define HASH_SIZE (1 << HASH_BITS)

static double now(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* Номер ячейки в head для трёх байт: сдвиги и xor, результат 0..HASH_SIZE-1. */
static int hash3(const uint8_t *p)
{
    return ((p[0] << 10) ^ (p[1] << 5) ^ p[2]) & (HASH_SIZE - 1);
}

static int compress_block(const uint8_t *in, int n, uint8_t *out)
{
    int *head = malloc(HASH_SIZE * sizeof(int)); /* последняя позиция с этим хэшем */
    int *prev = malloc(n * sizeof(int));         /* предыдущая позиция с тем же хэшем */
    for (int h = 0; h < HASH_SIZE; h++)
        head[h] = -1;

    int o = 0;
    int i = 0;
    while (i < n) {
        int best_len = 0;
        int best_dist = 0;

        if (i + MIN_MATCH <= n) {
            int cand = head[hash3(in + i)];
            for (int c = 0; cand >= 0 && c < MAX_CHAIN && i - cand <= WINDOW; c++) {
                int len = 0;
                while (len < MAX_MATCH && i + len < n && in[cand + len] == in[i + len])
                    len++;
                if (len > best_len) {
                    best_len = len;
                    best_dist = i - cand;
                }
                cand = prev[cand];
            }
        }

        int step;
        if (best_len >= MIN_MATCH) {
            out[o++] = 1;
            out[o++] = best_dist & 255;
            out[o++] = best_dist >> 8;
            out[o++] = best_len - MIN_MATCH;
            step = best_len;
        } else {
            out[o++] = 0;
            out[o++] = in[i];
            step = 1;
        }

        for (int k = 0; k < step; k++) {
            int p = i + k;
            if (p + MIN_MATCH <= n) {
                int h = hash3(in + p);
                prev[p] = head[h];
                head[h] = p;
            }
        }
        i += step;
    }

    free(head);
    free(prev);
    return o;
}

static void decompress_block(const uint8_t *in, int csize, uint8_t *out)
{
    int i = 0;
    int o = 0;
    while (i < csize) {
        if (in[i++] == 0) {
            out[o++] = in[i++];
        } else {
            int dist = in[i] | (in[i + 1] << 8);
            int len = in[i + 2] + MIN_MATCH;
            i += 3;
            for (int k = 0; k < len; k++) { /* match может перекрываться с выводом */
                out[o] = out[o - dist];
                o++;
            }
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
    if (fread(buf, 1, *size, f) != *size) {
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

    uint8_t **res = malloc((nblocks + 1) * sizeof(uint8_t *));
    int *csize = malloc((nblocks + 1) * sizeof(int));

    double t0 = now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)nblocks; b++) {
        size_t off = (size_t)b * BLOCK;
        int n = (int)(size - off < BLOCK ? size - off : BLOCK);
        res[b] = malloc(2 * (size_t)n + 16);
        csize[b] = compress_block(in + off, n, res[b]);
    }

    double t1 = now();

    /* запись строго по порядку блоков */
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

    size_t size, nblocks;
    memcpy(&size, in, sizeof(size_t));
    memcpy(&nblocks, in + sizeof(size_t), sizeof(size_t));
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
    for (long b = 0; b < (long)nblocks; b++)
        decompress_block(in + offs[b], csize[b], out + (size_t)b * BLOCK);

    double t1 = now();

    FILE *f = fopen(outpath, "wb");
    if (!f) {
        perror(outpath);
        exit(1);
    }
    fwrite(out, 1, size, f);
    fclose(f);

    fprintf(stderr, "threads=%d compute=%.4f s\n", threads, t1 - t0);
    free(offs);
    free(csize);
    free(out);
    free(in);
}

/* Возвращает 0, если распакованные данные совпали с оригиналом. */
static int verify_file(const char *inpath, int threads)
{
    size_t size;
    uint8_t *in = read_file(inpath, &size);
    size_t nblocks = (size + BLOCK - 1) / BLOCK;

    uint8_t **res = malloc((nblocks + 1) * sizeof(uint8_t *));
    int *csize = malloc((nblocks + 1) * sizeof(int));
    uint8_t *out = malloc(size + 1);

    double t0 = now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)nblocks; b++) {
        size_t off = (size_t)b * BLOCK;
        int n = (int)(size - off < BLOCK ? size - off : BLOCK);
        res[b] = malloc(2 * (size_t)n + 16);
        csize[b] = compress_block(in + off, n, res[b]);
    }

    double t1 = now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (long b = 0; b < (long)nblocks; b++)
        decompress_block(res[b], csize[b], out + (size_t)b * BLOCK);

    double t2 = now();

    size_t total = 0;
    for (size_t b = 0; b < nblocks; b++) {
        total += csize[b];
        free(res[b]);
    }

    int ok = memcmp(in, out, size) == 0;
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
