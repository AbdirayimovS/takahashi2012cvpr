import numpy as np
import tnm

def test_householder():
    n = np.random.rand(3)
    n = n / np.linalg.norm(n)
    if n[2] > 0:
        n = -n
    d = np.random.rand()

    N = 100
    p = np.random.rand(N*3).reshape((N, 3))

    H = tnm.householder(n, d)
    q = H[:3, :3] @ p.T + H[:3, 3].reshape((3,1))
    q = q.T

    diff = p - q
    diff = diff / np.linalg.norm(diff, axis=1).reshape((-1, 1))

    print(n.T @ diff.T)

    p2 = H[:3, :3] @ q.T + H[:3, 3].reshape((3,1))
    p2 = p2.T

    assert np.allclose(p, p2)
    
    print('OK')

def test_sph():
    n = np.random.rand(300).reshape((100, 3)) - 0.5
    n = n / np.linalg.norm(n, axis=1).reshape((-1, 1))
    theta, phi = tnm.cart2sph(n)
    n2 = tnm.sph2cart(theta, phi)
    assert np.allclose(n, n2)

    print('OK')

test_sph()