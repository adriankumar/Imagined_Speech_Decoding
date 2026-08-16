#two spaces: electrode (isolates sh loss) and image (sh + interpolation (representative for the model's view))
#losses -> lower is better, 0 is perfect | scores -> higher is better, 1 is perfect

from .electrode_space import (difference, sqr_diff_ratio, mean_error, 
                              recovered_variance, cosine_sim)

#sobel stack for visual, otherwise loss computes internally
from .image_space import (pixel_loss, sobel_stack, sobel_loss)