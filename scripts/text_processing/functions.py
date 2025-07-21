def is_two_column_layout(page):
    """single- or double-column text blocks"""
    text_blocks = page.get_text("blocks")
    x_coords = [block[0] for block in text_blocks]

    # Cluster x-coordinates to determine if we have two columns
    left_x = [x for x in x_coords if x < page.rect.width / 2]
    right_x = [x for x in x_coords if x >= page.rect.width / 2]

    # If we have significant separation in x-coordinates = two columns
    return len(left_x) > 0 and len(right_x) > 0


def extract_text_from_single_column(page):
    """Extract text line by line for single-column layout."""
    text = page.get_text("text")
    return text


def extract_text_from_two_columns(page):
    """Extract text by splitting page into two columns."""

    # Two Section Split
    mid_x = page.rect.width / 2
    text_blocks = page.get_text("blocks")

    # Text blocks into left and right columns Separation
    left_blocks = [block for block in text_blocks if block[0] < mid_x]
    right_blocks = [block for block in text_blocks if block[0] >= mid_x]

    # maintain reading orderbase Y-coordinate
    left_blocks = sorted(left_blocks, key=lambda b: b[1])
    right_blocks = sorted(right_blocks, key=lambda b: b[1])

    # Concatenate text
    left_text = " ".join([block[4] for block in left_blocks])
    right_text = " ".join([block[4] for block in right_blocks])

    # Merge left and right columns line by line
    combined_text = "\n".join([left_text, right_text])
    return combined_text