import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import tensorflow as tf

from . import loss


def assign_bin_numbers(x: np.array, bin_centers: np.array):
    x = x.reshape(-1, 1)
    bin_centers = bin_centers.reshape(1, bin_centers.size)
    diff = np.square(x - bin_centers)
    bins = np.argmin(diff, -1)
    return bins


def test_nmi_abstract():
    nmi_loss = loss.MutualInformation(
        n_bins=4,
        min_val=0,
        max_val=3,
        normalize=True,
        clip=True,
        include_endpoints=True,
        debug=True,
        sigma_ratio=1e-5,
    )
    assert np.allclose(nmi_loss.bin_centers, np.arange(4))
    to_tensor = lambda x: tf.convert_to_tensor(np.array(x).reshape(1, -1), dtype=tf.float32)
    no_mi = nmi_loss(to_tensor([0, 0, 0, 0]), to_tensor([0, 1, 2, 3]))
    assert np.isclose(0, no_mi, atol=1e-5)
    high_mi = nmi_loss(to_tensor([0, 0, 1, 1]), to_tensor([0, 0, 1, 1]))
    assert np.isclose(1, high_mi, atol=1e-5)
    return


def test_nmi_images():
    """
    Test the mutual information and compare it with the sklearn implementation.
    It is not implemented with the differentiable method, but with low sigma
    ration, it is similar.
    """
    nmi_loss = loss.MutualInformation(
        n_bins=100,
        min_val=0,
        max_val=1,
        normalize=True,
        clip=True,
        debug=True,
        sigma_ratio=1e-1,  # then it is the same as in sk-learn, but lower creates too high numbers
    )

    to_tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

    img_shape = (1, 64, 64, 1)
    n_voxels = np.prod(img_shape)
    regular_img = np.linspace(0, 1, n_voxels).reshape(img_shape)
    regular_img_2 = np.linspace(1, 0, n_voxels).reshape(img_shape)

    mi_tf = nmi_loss(to_tensor(regular_img), to_tensor(regular_img_2))
    mi_sk = normalized_mutual_info_score(
        assign_bin_numbers(regular_img, nmi_loss.bin_centers.numpy()),
        assign_bin_numbers(regular_img_2, nmi_loss.bin_centers.numpy()),
    )
    np.allclose(mi_tf, mi_sk, atol=0.03)

    rand_img = np.random.rand(n_voxels).reshape(img_shape)
    rand_img_2 = np.random.rand(n_voxels).reshape(img_shape)
    higher = nmi_loss(to_tensor(rand_img), to_tensor(rand_img + 0.2))
    assert not tf.math.is_nan(higher), "Result should be finite"
    diff = nmi_loss(to_tensor(rand_img), to_tensor(rand_img_2))
    diff_switched = nmi_loss(to_tensor(rand_img_2), to_tensor(rand_img))
    assert np.allclose(diff, diff_switched, atol=1e-3), "NMI should be symmetric"
    diff_sk = normalized_mutual_info_score(
        assign_bin_numbers(rand_img, nmi_loss.bin_centers.numpy()),
        assign_bin_numbers(rand_img_2, nmi_loss.bin_centers.numpy()),
    )
    assert np.allclose(diff, diff_sk, atol=0.03)
    return


def test_constraint_output():
    """Test the constraint output by just calculating a few values"""
    standard_loss = loss.ConstrainOutput(reduction="none")
    test_numbers = tf.convert_to_tensor([-0.3, -0.1, 0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 10])
    target_numbers = np.array([0.3, 0.1, 0, 0, 0, 0, 0.2, 0.5, 9])
    loss_vals = standard_loss(None, test_numbers).numpy()
    assert np.allclose(loss_vals, target_numbers)
    # do the whole thing with scaling
    scaled_loss = loss.ConstrainOutput(reduction="none", scaling=2)
    loss_vals = scaled_loss(None, test_numbers).numpy()
    assert np.allclose(loss_vals, 2 * target_numbers)
    # try different min and max values
    range_loss = loss.ConstrainOutput(reduction="none", min_val=-1, max_val=2)
    test_numbers = tf.convert_to_tensor([-1.3, -1.1, 0.1, 0.5, 0.8, 1.0, 2.2, 2.5, 10])
    target_numbers = np.array([0.3, 0.1, 0, 0, 0, 0, 0.2, 0.5, 8])
    loss_vals = range_loss(None, test_numbers).numpy()
    assert np.allclose(loss_vals, target_numbers)
