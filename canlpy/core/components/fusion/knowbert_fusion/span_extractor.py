import torch
import torch.nn as nn

class SpanExtractor(nn.Module):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.

    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """

    def forward(self, # pylint: disable=arguments-differ
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None):
        """
        Given a sequence tensor, extract spans and return representations of
        them. Span representation can be computed in many different ways,
        such as concatenation of the start and end spans, attention over the
        vectors contained inside the span, etc.

        Parameters
        ----------
        sequence_tensor : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : ``torch.LongTensor``, required.
            A tensor of shape ``(batch_size, num_spans, 2)``, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the ``sequence_tensor``.
        sequence_mask : ``torch.LongTensor``, optional (default = ``None``).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : ``torch.LongTensor``, optional (default = ``None``).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the ``indices`` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.

        Returns
        -------
        A tensor of shape ``(batch_size, num_spans, embedded_span_size)``,
        where ``embedded_span_size`` depends on the way spans are represented.
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        """
        Returns the expected final dimension of the ``sequence_tensor``.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the expected final dimension of the returned span representation.
        """
        raise NotImplementedError

class SelfAttentiveSpanExtractor(SpanExtractor):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.

    Returns
    -------
    attended_text_embeddings : ``torch.FloatTensor``.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """
    def __init__(self,
                 input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = torch.nn.Linear(input_dim, 1)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:

        dtype = sequence_tensor.dtype

        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # Shape: (1, 1, max_batch_span_width) eg: [[[0,1]]]
        max_span_range_indices = torch.arange(max_batch_span_width, device = sequence_tensor.device).view(1, 1, -1)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).to(dtype)
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).to(dtype)
        span_indices = torch.nn.functional.relu(raw_span_indices.to(dtype)).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = batched_index_select(global_attention_logits,
                                                          span_indices,
                                                          flat_span_indices).squeeze(-1)
        # print(span_attention_logits.shape)
        # print(span_mask.shape)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = masked_softmax(span_attention_logits, span_mask,
                                                     memory_efficient=True,
                                                     mask_fill_value=-1000)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)

        #Do weighted sum of embeddings corresponding to same span
        attended_text_embeddings = (span_embeddings*span_attention_weights.unsqueeze(dim=-1)).sum(dim=-2)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).to(dtype)

        return attended_text_embeddings

def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for :func:`~batched_index_select`. The given ``indices`` of size
    ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor, which has size
    ``(batch_size, sequence_length, embedding_size)``. This function returns a vector that
    correctly indexes into the flattened target. The sequence length of the target must be
    provided to compute the appropriate offsets.

    .. code-block:: python

        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]

    Parameters
    ----------
    indices : ``torch.LongTensor``, required.
    sequence_length : ``int``, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    Returns
    -------
    offset_indices : ``torch.LongTensor``
    """
    # Shape: (batch_size)
    offsets = torch.arange(indices.size(0), device=indices.device) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices

def batched_index_select(target: torch.Tensor,
                        indices: torch.LongTensor,
                        flattened_indices: torch.LongTensor = None) -> torch.Tensor:
    """
    The given ``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into the sequence
    dimension (dimension 2) of the target, which has size ``(batch_size, sequence_length,
    embedding_size)``.

    This function returns selected values in the target with respect to the provided indices, which
    have size ``(batch_size, d_1, ..., d_n, embedding_size)``. This can use the optionally
    precomputed :func:`~flattened_indices` with size ``(batch_size * d_1 * ... * d_n)`` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    :class:`~allennlp.models.coreference_resolution.CoreferenceResolver`. Model to select
    contextual word representations corresponding to the start and end indices of mentions. The key
    reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    Parameters
    ----------
    target : ``torch.Tensor``, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : ``torch.LongTensor``
        A tensor of shape (batch_size, ...), where each element is an index into the
        ``sequence_length`` dimension of the ``target`` tensor.
    flattened_indices : Optional[torch.Tensor], optional (default = None)
        An optional tensor representing the result of calling :func:~`flatten_and_batch_shift_indices`
        on ``indices``. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    Returns
    -------
    selected_targets : ``torch.Tensor``
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        dtype = vector.dtype
        mask = mask.to(dtype)
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result