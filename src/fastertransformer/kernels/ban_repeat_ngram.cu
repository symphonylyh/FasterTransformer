/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/kernels/ban_repeat_ngram.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
__global__ void ban_repeat_ngram(T*          logits,
                                 const int*  output_ids_buf,
                                 const bool* finished_buf,
                                 const int*  parent_ids_buf,
                                 int         batch_size,
                                 int         beam_width,
                                 int         no_repeat_ngram_size,
                                 int         id_offset,
                                 int         vocab_size_padded,
                                 size_t      step)
{
    /**
     * Find subsequences that match the last (ngram_size - 1) generated tokens. The next-tokens of those matching
     * subsequences should be banned from the next-token logits, such that existing N-grams in current generated
     * sequence will not be generated again.
     *
     * Note 1: no-repeat restriction is per-beam instead of across-beam.
     * Note 2: since beam search results are stored and retrieved by backtracking (parent_ids), the entire sequence for
     * the current can only be obtained by backtracking all the way back.
     * Note 3: for greedy search, actually a more efficient implementaion can be adopted (since we're not constrained by
     * the beam backtrack retrival, we can have all threads loading into shared mem in parallel AND use normal order
     * traversal). But for simplicity and consistency, greedy search and beam search implementation are kept the same.
     *
     * The overlap between adjacent threads indicates wasted global memory access. Used shared memory instead.
     * Shared memory benefit is more significant as ngram_size increases. Extra elementsShared memory reuse is for
     * in-bound positions only. For leftside out-of-boundary tokens, access by global memory.
     */

    const int  output_idx      = blockIdx.x * blockDim.x + threadIdx.x;
    const int  local_batch_idx = blockIdx.y / beam_width;
    const int  beam_idx        = blockIdx.y % beam_width;
    const bool beam_search     = beam_width > 1;

    // if the beam has already finished, skip ngram check
    if (finished_buf[id_offset + local_batch_idx * beam_width + beam_idx])
        return;

    // shared mem: one token for each thread, plus (ngram_size - 1) extra tokens beyond block boundary, plus (ngram_size
    // - 1) most recent tokens
    extern __shared__ int shared_tokens[];
    int                   shared_tokens_length = blockDim.x + no_repeat_ngram_size - 1;
    int*                  last_tokens          = &shared_tokens[shared_tokens_length];
    int                   last_tokens_length   = no_repeat_ngram_size - 1;

    // retrive the entire beam by backtracking from last token to current token  (in reverse order)
    // single thread vs parallel thread is equivalent as it's bound by the longest iteration
    if (threadIdx.x == 0) {
        int parent_id        = beam_idx;
        int start_record_idx = min(output_idx + shared_tokens_length, (int)step);
        int shared_token_idx = start_record_idx == step ? step - output_idx - 1 : shared_tokens_length - 1;
        int last_token_idx   = last_tokens_length - 1;
        // write to shared mem in reverse order; boundary condition when thread block covers more than step

        for (int curr_idx = step - 1; curr_idx >= output_idx; curr_idx--) {
            if (last_token_idx >= 0)
                last_tokens[last_token_idx--] = output_ids_buf[curr_idx * batch_size * beam_width + id_offset
                                                               + local_batch_idx * beam_width + parent_id];

            // before reaching the part of current block, traverse only; after that, record the tokens
            if (curr_idx < start_record_idx)
                shared_tokens[shared_token_idx--] = output_ids_buf[curr_idx * batch_size * beam_width + id_offset
                                                                   + local_batch_idx * beam_width + parent_id];

            if (beam_search)
                parent_id = parent_ids_buf[curr_idx * batch_size * beam_width + id_offset + local_batch_idx * beam_width
                                           + parent_id];
        }
    }

    __syncthreads();

    if (output_idx > step - no_repeat_ngram_size)
        return;

    bool ban_ngram = true;

    // ngram check (in regular order)
    for (int ngram_idx = 0; ngram_idx < no_repeat_ngram_size - 1; ngram_idx++) {
        if (shared_tokens[threadIdx.x + ngram_idx] != last_tokens[ngram_idx]) {
            ban_ngram = false;
            break;
        }
    }

    if (ban_ngram) {
        int banned_token = shared_tokens[threadIdx.x + no_repeat_ngram_size - 1];  // ban the last token in the ngram
        logits[local_batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token] =
            static_cast<T>(-INFINITY);  // note: "logits" passed in is already with batchxbeam offset
        // printf("[DEBUG] step %d, beam %d: ban ngram ...[%d, %d, %d]\n",
        //        (int)step,
        //        (int)beam_idx,
        //        (int)shared_tokens[threadIdx.x + no_repeat_ngram_size - 1 - 2],
        //        (int)shared_tokens[threadIdx.x + no_repeat_ngram_size - 1 - 1],
        //        banned_token);
    }
}

template<typename T>
void invokeBanRepeatNgram(T*           logits,
                          const int*   output_ids_buf,
                          const bool*  finished_buf,
                          const int*   parent_ids_buf,
                          int          batch_size,
                          int          local_batch_size,
                          int          beam_width,
                          int          no_repeat_ngram_size,
                          int          id_offset,
                          int          vocab_size_padded,
                          size_t       step,
                          cudaStream_t stream)
{

    // case 1: ngram_size == 0 --> this means no ngram limit
    // case 2: generated length must be greater than ngram_size to do ngram check
    if (no_repeat_ngram_size == 0 || step < no_repeat_ngram_size)
        return;

    // step (current generated length, except start token) is from 1 ~ max_seq_len
    dim3 block, grid;
    block.x = min(((step + 32 - 1) / 32) * 32, 256UL);
    grid.x  = (step + block.x - 1) / block.x;
    grid.y  = local_batch_size * beam_width;

    // dynamically allocate shared memory of int[blockDim + 2*(ngram_size - 1)], where ngram_size - 1 is for boundary
    // token's ngram and for most recent tokens
    ban_repeat_ngram<<<grid, block, (block.x + 2 * (no_repeat_ngram_size - 1)) * sizeof(int), stream>>>(
        logits,
        output_ids_buf,
        finished_buf,
        parent_ids_buf,
        batch_size,
        beam_width,
        no_repeat_ngram_size,
        id_offset,
        vocab_size_padded,
        step);
    sync_check_cuda_error();
}

template void invokeBanRepeatNgram(half*        logits,
                                   const int*   output_ids_buf,
                                   const bool*  finished_buf,
                                   const int*   parent_ids_buf,
                                   int          batch_size,
                                   int          local_batch_size,
                                   int          beam_width,
                                   int          no_repeat_ngram_size,
                                   int          id_offset,
                                   int          vocab_size_padded,
                                   size_t       step,
                                   cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBanRepeatNgram(__nv_bfloat16* logits,
                                   const int*     output_ids_buf,
                                   const bool*    finished_buf,
                                   const int*     parent_ids_buf,
                                   int            batch_size,
                                   int            local_batch_size,
                                   int            beam_width,
                                   int            no_repeat_ngram_size,
                                   int            id_offset,
                                   int            vocab_size_padded,
                                   size_t         step,
                                   cudaStream_t   stream);
#endif
template void invokeBanRepeatNgram(float*       logits,
                                   const int*   output_ids_buf,
                                   const bool*  finished_buf,
                                   const int*   parent_ids_buf,
                                   int          batch_size,
                                   int          local_batch_size,
                                   int          beam_width,
                                   int          no_repeat_ngram_size,
                                   int          id_offset,
                                   int          vocab_size_padded,
                                   size_t       step,
                                   cudaStream_t stream);

}  // namespace fastertransformer
