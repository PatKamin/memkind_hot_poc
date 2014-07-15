/*
 * Copyright (C) 2014 Intel Corperation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice(s),
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice(s),
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef numakind_include_h
#define numakind_include_h
#ifdef __cplusplus
extern "C" {
#endif

/*!
 *  \mainpage
 *
 *  The numakind library extends libnuma with the ability to
 *  categorize groups of numa nodes into different "kinds" of
 *  memory. It provides a low level interface for generating inputs to
 *  mbind() and mmap(), and a high level interface for heap
 *  management.  The heap management is implemented with an extension
 *  to the jemalloc library which dedicates "arenas" to each CPU and
 *  kind of memory.  To use numakind, jemalloc must be compiled with
 *  the --enable-numakind option.

 */


typedef enum {
    NUMAKIND_DEFAULT = 0,
    NUMAKIND_HBW,
    NUMAKIND_HBW_HUGETLB,
    NUMAKIND_HBW_PREFERRED,
    NUMAKIND_HBW_PREFERRED_HUGETLB,
    NUMAKIND_NUM_KIND
} numakind_t;

typedef enum {
    NUMAKIND_ERROR_UNAVAILABLE = -1,
    NUMAKIND_ERROR_MBIND = -2,
    NUMAKIND_ERROR_MEMALIGN = -3,
    NUMAKIND_ERROR_MALLCTL = -4,
    NUMAKIND_ERROR_MALLOC = -5,
    NUMAKIND_ERROR_GETCPU = -6,
    NUMAKIND_ERROR_HBW = -7,
    NUMAKIND_ERROR_PMTT = -8,
    NUMAKIND_ERROR_TIEDISTANCE = -9,
    NUMAKIND_ERROR_ALIGNMENT = -10,
    NUMAKIND_ERROR_ALLOCM = -11,
    NUMAKIND_ERROR_ENVIRON = -12
} numakind_error_t;

static const size_t NUMAKIND_ERROR_MESSAGE_SIZE = 128;

/* Convert error number into an error message */
void numakind_error_message(int err, char *msg, size_t size);

/* returns 1 if numa kind is availble else 0 */
int numakind_is_available(int kind);

/* sets nodemask for the nearest numa node for the specified numa kind */
int numakind_get_nodemask(int kind, unsigned long *nodemask, unsigned long maxnode);

/* set flags for call to mmap */
int numakind_get_mmap_flags(int kind, int *flags);

/* mbind to the nearest numa node of the specified kind */
int numakind_mbind(int kind, void *addr, size_t len);

/* malloc from the nearest numa node of the specified kind */
void *numakind_malloc(numakind_t kind, size_t size);

/* calloc from the nearest numa node of the specified kind */
void *numakind_calloc(numakind_t kind, size_t num, size_t size);

/* posix_memalign from the nearest numa node of the specified kind */
int numakind_posix_memalign(numakind_t kind, void **memptr, size_t alignment,
                            size_t size);

/* realloc from the nearest numa node of the specified kind */
void *numakind_realloc(numakind_t kind, void *ptr, size_t size);

/* Free memory allocated with the numakind API */
void numakind_free(numakind_t kind, void *ptr);

#ifdef __cplusplus
}
#endif
#endif
