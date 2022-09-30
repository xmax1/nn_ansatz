def generate_real_lattice(real_basis, rcut, reciprocal_height):
    # from pyscf, some logic to set the number of imgs away from simulation cell. Adapted version confirmed in notion
    nimgs = jnp.ceil(rcut*reciprocal_height + 1.1).astype(int)
    img_range = jnp.arange(-nimgs, nimgs+1)
    img_sets = list(product(*[img_range, img_range, img_range]))
    # first axis is the number of lattice vectors, second is the integers to scale the primitive vectors, third is the resulting set of vectors
    # then sum over those
    # print(len(img_sets))
    img_sets = jnp.concatenate([jnp.array(x)[None, :, None] for x in img_sets if not jnp.sum(jnp.array(x) == 0) == 3], axis=0)
    # print(img_sets.shape)
    imgs = jnp.sum(img_sets * real_basis, axis=1)
    
    # generate all the single combinations of the basis vectors
    v = jnp.split(real_basis, 3, axis=0)
    z = jnp.zeros_like(v[0])
    vecs = product(*[[-v[0], z, v[0]],[-v[1], z, v[1]], [-v[2], z, v[2]]])
    vecs = jnp.array(list(vecs)).squeeze().sum(-2)  # sphere around the origin

    # if a sphere around the image is within rcut then keep it
    lengths = jnp.linalg.norm(vecs[None, ...] + imgs[:, None, :], axis=-1)
    mask = jnp.any(lengths < rcut, axis=1)
    nimgs = len(imgs)
    imgs = imgs[mask]
    return imgs


def generate_reciprocal_lattice(reciprocal_basis, mesh):
    # 3D uniform grids
    rx = jnp.fft.fftfreq(mesh[0], 1./mesh[0])
    ry = jnp.fft.fftfreq(mesh[1], 1./mesh[1])
    rz = jnp.fft.fftfreq(mesh[2], 1./mesh[2])
    base = (rx, ry, rz)
    cartesian_product = jnp.array(list(product(*base)))  # another worse version of this is available in cartesian_prod(...)
    cartesian_product = jnp.array([x for x in cartesian_product if not jnp.sum(x == 0) == 3])  # filter the zero vector
    reciprocal_lattice = jnp.dot(cartesian_product, reciprocal_basis)
    return reciprocal_lattice