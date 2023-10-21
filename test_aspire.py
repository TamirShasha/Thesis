if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from aspire.basis import FFBBasis2D
    from aspire.utils.misc import grid_2d
    from scipy.special import jv

    L = 5
    b = FFBBasis2D(size=(L, L), dtype=float)

    coef_ref = np.zeros(b.count, dtype=float)
    coef_ref[0] = 1
    a1 = b.evaluate(coef_ref).asnumpy()[0]

    coef_ref = np.zeros(b.count, dtype=float)
    coef_ref[2] = 1
    a2 = b.evaluate(coef_ref).asnumpy()[0]

    print(b.count)
    print(a1.shape)
    plt.imshow(a1, cmap='gray')
    plt.show()

    plt.imshow(a2, cmap='gray')
    plt.show()

    print(np.sum(a1 * a2))

    # coef = b.expand(im)
