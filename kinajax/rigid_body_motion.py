import jax
from jax import jit, numpy as jnp
from jax.lax import cond, concatenate

## CONDITIONAL HELPER FUNCTIONS
def norm_true_func(x):
    dot,v1,v2 = x
    return dot
def norm_false_func(x):
    dot, v1, v2 = x
    return dot / jnp.linalg.norm(v1, axis=1) / jnp.linalg.norm(v2, axis=1)
def deg_true_func(angles):
    return jnp.degrees(angles)
def deg_false_func(angles):
    return angles

@jit
def tan_angle(ref_v,v1,v2, normalized : bool, deg : bool):
    """
    Computes the angle between two vectors relative to a plane defined by it's normal

    Parameters
    ----------
    ref_v : array_like
        3D unit vector corresponding to the normal of the reference plane
    v1, v2: 
        3D vectors of the 2 vectors to find the angle between
    normalized : bool
        If True, then v1 and v2 vectors are unit vecotr.  If False, v1 and v2 need to be normalized
    deg: bool
        If true, return angles in degrees (Note, this may cause problems with the omega function)

    Raises
    ------
    ValueError
        If v1 and v2 do not have the same number of dimension
        If ref_v, v1, and v2 are all not the same shape

    Example Call
    ------------
    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> x = jax.random.normal(key_x,shape=(100,3))
    >>> y = jax.random.normal(key_y,shape=(100,3))
    >>> z = jax.random.normal(key_z,shape=(100,3))

    >>> x = x / jnp.linalg.norm(x, axis=1)[:,None]
    >>> y = y / jnp.linalg.norm(y, axis=1)[:,None]
    >>> z = z / jnp.linalg.norm(z, axis=1)[:,None]

    >>> theta = tan_angle(z, x, y, normalized=True, deg=False)

    """
    if v1.ndim != v2.ndim:
        raise ValueError("Dimensions of V1 and V2 are not equal")
    
    if ref_v.shape != v1.shape:
        raise ValueError("Reference vector is not he same shape as v1")
    
    if ref_v.shape != v2.shape:
        raise ValueError("Reference vector is not he same shape as v2")
    
    if v1.ndim == 1:
        det = jnp.dot(ref_v,jnp.cross(v1,v2))
        dot = jnp.dot(v1, v2)
    else:  
        det = jnp.sum(jnp.multiply(ref_v,jnp.cross(v1,v2)), axis=1)
        dot = jnp.sum(jnp.multiply(v1,v2), axis=1)

    dot = cond(normalized, norm_true_func, norm_false_func, (dot,v1,v2))
    angles = jnp.arctan2(det,dot)
    return cond(deg, deg_true_func, deg_false_func, angles)
        
@jit
def cos_angle(v1, v2, normalized : bool, deg : bool):
    """
    Computes the cosine angle betweeen two planes defined by their normal's

    Parameters
    ----------
    v1, v2: 
        3D vectors of the 2 vectors to find the angle between
    normalized : bool
        If True, then v1 and v2 vectors are unit vecotr.  If False, v1 and v2 need to be normalized
    deg: bool
        If true, return angles in degrees (Note, this may cause problems with the omega function)

    Raises
    ------
    ValueError
        If v1, and v2 are not the same shape

    Example Call
    ------------

    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> x = jax.random.normal(key_x,shape=(100,3))
    >>> y = jax.random.normal(key_y,shape=(100,3))
    >>> z = jax.random.normal(key_z,shape=(100,3))

    >>> x = x / jnp.linalg.norm(x, axis=1)[:,None]
    >>> y = y / jnp.linalg.norm(y, axis=1)[:,None]
    >>> z = z / jnp.linalg.norm(z, axis=1)[:,None]

    >>> theta = cos_angle(z, x, y, normalized=True, deg=False)

    """
    if v1.shape != v2.shape:
        raise ValueError("Reference vector is not he same shape as v1")

    if v1.ndim == 1:
        dot_prod = jnp.dot(v1, v2)
    else:
        dot_prod = jnp.sum(jnp.multiply(v1, v2), axis=1)

    cosines = cond(normalized, norm_true_func, norm_false_func, (dot_prod, v1,v2))

    angles = jnp.arccos(jnp.clip(cosines, -1, 1))

    return cond(deg, deg_true_func, deg_false_func, angles)

###########################################################
### DO I NEED GET COSINE AND ALT ANGLES FUNCTIONS?
#####################################################

@jit
def get_omega_angles(theta, phi, psi, dt):
    """
    Computes the angular velocities from theta, phi, and psi

    Parameters:
    -----------
    theta, phi, psi : array_like or float
        euler angles of the rigid body
    dt : array_like or float
        the time interval to calcuate change in position relative to

    Raises
    ------
        If dt is an array and not the same length as theta, phi, and psi

    Returns
    -------
    omega : tuple
        angular velocities around the 3 principle axis of the rigid body

    Example Call
    ------------
    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> x = jax.random.normal(key_x,shape=(100,3))
    >>> y = jax.random.normal(key_y,shape=(100,3))
    >>> z = jax.random.normal(key_z,shape=(100,3))

    >>> x = x / jnp.linalg.norm(x, axis=1)[:,None]
    >>> y = y / jnp.linalg.norm(y, axis=1)[:,None]
    >>> z = z / jnp.linalg.norm(z, axis=1)[:,None]
    >>> delta_time = 1/300

    >>> theta = cos_angle(x,y,True,False)
    >>> phi = cos_angle(x,z,True, False)
    >>> psi = cos_angle(z,y,True, False)

    >>> omega = get_omega(theta, phi, psi, delta_time)
    """

    if (isinstance(dt, jnp.ndarray)):
        if dt.shape != theta.shape:
            raise ValueError(f"If dt is an array, it must be shape {theta.shape}")
    
    omega_one = jnp.gradient(theta) / dt
    omega_two = (jnp.gradient(phi) / dt) * jnp.sin(theta)
    omega_three = (jnp.gradient(phi) / dt) + (jnp.gradient(psi) / dt)

    return (omega_one, omega_two, omega_three)

@jit
def get_omega_R(R, dt):
    """
    Computes the angular velocities from the derivative of the rotation matrix

    Parameters:
    -----------
    R : array_like
        rotation matrix to map the local coordinate system back to the non-inertial frame
    dt : array_like or float
        the time interval to calcuate change in position relative to

    Raises
    ------
        If dt is an array and not the same length as theta, phi, and psi

    Returns
    -------
    omega : tuple
        angular velocities around the 3 principle axis of the rigid body

    Example Call
    ------------
    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> x = jax.random.normal(key_x,shape=(100,3))
    >>> y = jax.random.normal(key_y,shape=(100,3))
    >>> z = jax.random.normal(key_z,shape=(100,3))

    >>> x = x / jnp.linalg.norm(x, axis=1)[:,None]
    >>> y = y / jnp.linalg.norm(y, axis=1)[:,None]
    >>> z = z / jnp.linalg.norm(z, axis=1)[:,None]
    >>> delta_time = 1/300

    >>> R = np.stack([x,y,z],axis=2)

    >>> omega = get_omega(R, delta_time)
    """

    # if (isinstance(dt, jnp.ndarray)):
    #     if dt.shape[0]!= R.shape[0]:
    #         raise ValueError(f"If dt is an array, it must be length {R.shape[0]}")
    
    dRdt = jnp.diff(R, axis=0) / dt

    # Append a zero matrix at the beginning
    zero_matrix = jnp.zeros_like(R[:1])  # Shape (1, 3, 3)
    dRdt = jnp.vstack((zero_matrix, dRdt))
    Rt = jnp.transpose(R, (0,2,1))
    omega_matrices = jnp.einsum('ijk,ikl->ijl', dRdt, Rt)
    
    omega_one = -omega_matrices[:, 1, 2]
    omega_two = omega_matrices[:, 0, 2]
    omega_three = -omega_matrices[:, 0, 1]

    return (omega_one, omega_two, omega_three)

@jit
def ang_momentum(omega_one, omega_two, omega_three, I):
    """
    Computes the angular momentum from the components of omega, and the inertia tensor given

    Parameters
    ----------
    omega_one, omega_two, omega_three : array_like
        angular velocities around the 3 principle axis of the rigid body
    I: array_like
        Either a (3,3) or (3,) array corresponding to the moments of inertia on the x, y, and z principle axis respectively

    Returns
    -------
    LangX, LangY, LangZ, Lang : array_like
        angular momentum values around the x, y, and z principle axis as well as the total angular momenum

    Example Call
    ------------
    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> x = jax.random.normal(key_x,shape=(100,3))
    >>> y = jax.random.normal(key_y,shape=(100,3))
    >>> z = jax.random.normal(key_z,shape=(100,3))

    >>> x = x / jnp.linalg.norm(x, axis=1)[:,None]
    >>> y = y / jnp.linalg.norm(y, axis=1)[:,None]
    >>> z = z / jnp.linalg.norm(z, axis=1)[:,None]
    >>> I = jnp.eye(3)
    >>> I = I.at[2,2].set(3)

    >>> delta_time = 1/300

    >>> theta = cos_angle(x,y,True,False)
    >>> phi = cos_angle(x,z,True, False)
    >>> psi = cos_angle(z,y,True, False)

    >>> omega = get_omega_angles(theta, phi, psi, delta_time)
    >>> ang_mom = ang_mometum(omega[0], omega[1], omega[2], I=tensor)

    """
    if I.shape == (3,3):
        I = jnp.diag(I)

    LangX = omega_one * I[0]
    LangY = omega_two * I[1]
    LangZ = omega_three * I[2]
    Lang = jnp.sqrt(LangX**2 + LangY**2 + LangZ**2)

    return (LangX, LangY, LangZ, Lang)


@jit
def ang_energy(omega_one, omega_two, omega_three, I):
    """
    Computes the angular energy from the components of omega, and the inertia tensor given

    Parameters
    ----------
    omega_one, omega_two, omega_three : array_like
        angular velocities around the 3 principle axis of the rigid body
    I: array_like
        Either a (3,3) or (3,) array corresponding to the moments of inertia on the x, y, and z principle axis respectively

    Returns
    -------
    Tang : array_like
        total angular energy of the rigid body

    Example Call
    ------------
    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> x = jax.random.normal(key_x,shape=(100,3))
    >>> y = jax.random.normal(key_y,shape=(100,3))
    >>> z = jax.random.normal(key_z,shape=(100,3))

    >>> x = x / jnp.linalg.norm(x, axis=1)[:,None]
    >>> y = y / jnp.linalg.norm(y, axis=1)[:,None]
    >>> z = z / jnp.linalg.norm(z, axis=1)[:,None]
    >>> I = jnp.eye(3)
    >>> I = I.at[2,2].set(3)

    >>> delta_time = 1/300

    >>> theta = cos_angle(x,y,True,False)
    >>> phi = cos_angle(x,z,True, False)
    >>> psi = cos_angle(z,y,True, False)

    >>> omega = get_omega(theta, phi, psi, delta_time)
    >>> ang_energy = ang_energy(omega[0], omega[1], omega[2], I=tensor)

    """
    if I.shape == (3,3):
        I = jnp.diag(I)

    Tang = (.5 * I[0] * omega_one**2) + (.5 * I[0] * omega_two**2) + (.5 * I[2] * omega_three**2)

    return Tang

@jit
def lin_momentum(com_x, com_y, com_z, m, dt):
    """
    Computes the linear momentum from the velocity of the center of mass and the mass of the rigid body

    Parameters
    ----------
    com_x, com_y, com_z : array_like
        coordinates for the center of mass of the rigid body
    m : float
        the mass of the rigid bod
    dt : array_like or float
        the time interval to calcuate change in position relative to
    
    Raises
    ------
        If dt is an array and not the same length as com_x

    Example Call
    ------------
    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> com_x = jax.random.normal(key_x,shape=(100,))
    >>> com_y = jax.random.normal(key_y,shape=(100,))
    >>> com_z = jax.random.normal(key_z,shape=(100,))

    mass = 4.5

    >>> delta_time = 1/300

    >>> lin_mom = lin_momentum(com_x, com_y, com_z, mass, delta_time)

    """
    if (len(dt.shape) > 1 ) and (dt.shape != com_x.shape):
        raise ValueError(f"dt should be a float or an array with shape {com_x.shape}")

    
    LlinX = m*(jnp.gradient(com_x) / dt)
    LlinY = m*(jnp.gradient(com_y) / dt)
    LlinZ = m*(jnp.gradient(com_z) / dt)
    Llin = jnp.sqrt(LlinX**2 + LlinY**2 + LlinZ**2)

    return (LlinX, LlinY, LlinZ, Llin)


@jit
def lin_energy(com_x, com_y, com_z, m, dt):
    """
    Computes the linear energy from the velocity of the center of mass and the mass of the rigid body

    Parameters
    ----------
    com_x, com_y, com_z : array_like
        coordinates for the center of mass of the rigid body
    m : float
        the mass of the rigid bod
    dt : array_like or float
        the time interval to calcuate change in position relative to
    
    Raises
    ------
        If dt is an array and not the same length as com_x

    Example Call
    ------------
    >>> key_x = jax.random.PRNGKey(133)
    >>> key_y = jax.random.PRNGKey(156)
    >>> key_z = jax.random.PRNGKey(120)
    >>> com_x = jax.random.normal(key_x,shape=(100,))
    >>> com_y = jax.random.normal(key_y,shape=(100,))
    >>> com_z = jax.random.normal(key_z,shape=(100,))

    mass = 4.5

    >>> delta_time = 1/300

    >>> lin_energy = lin_energy(com_x, com_y, com_z, mass, delta_time)

    """
    if (len(dt.shape) > 1 ) and (dt.shape != com_x.shape):
        raise ValueError(f"dt should be a float or an array with shape {com_x.shape}")

    
    vx = jnp.gradient(com_x) / dt
    vy = jnp.gradient(com_y) / dt
    vz = jnp.gradient(com_z) / dt
    v = jnp.sqrt(vx**2 + vy**2 + vz**2)

    return m*(v**2)

@jit
def get_kinetics(
        omega_one, 
        omega_two, 
        omega_three, 
        com_x, 
        com_y, 
        com_z, 
        I, 
        m, 
        dt
):
    """
    Computes the total momentum, total energy, and their components in one function

    Parameters
    ----------
    omega_one, omega_two, omega_three : array_like
        angular velocities around the 3 principle axis of the rigid body
    com_x, com_y, com_z : array_like
        coordinates for the center of mass of the rigid body
    I: array_like
        Either a (3,3) or (3,) array corresponding to the moments of inertia on the x, y, and z principle axis respectively
    m : float
        the mass of the rigid bod
    dt : array_like or float
        the time interval to calcuate change in position relative to

    Returns
    -------
    T, L : array_like
        two arrays that correspond to [angular_energy, linear energy, total_energy]
        and [ang_momX, ang_momY, ang_momZ, total_ang_mom, lin_momX, lin_momY, lin_momZ, total_lin_mom, total_mom]

    Example Call
    ------------
    # Genrate random data
    key_x = jax.random.PRNGKey(133)
    key_y = jax.random.PRNGKey(156)
    key_z = jax.random.PRNGKey(120)
    com_x = jax.random.normal(key_x,shape=(100,))
    com_y = jax.random.normal(key_y,shape=(100,))
    com_z = jax.random.normal(key_z,shape=(100,))
    theta = jax.random.normal(key_x,shape=(100,))
    phi = jax.random.normal(key_y,shape=(100,))
    psi = jax.random.normal(key_z,shape=(100,))

    I = jnp.eye(3)
    I = I.at[2,2].set(3)

    delta_time = 1/300
    mass = 4.5

    omega = get_omega(theta,phi,psi,delta_time)
    T, L = get_kinetics(
        omega_one=omega[0],
        omega_two=omega[1],
        omega_three=omega[2],
        com_x=x[:,0], 
        com_y=y[:,1], 
        com_z=z[:,2],
        I = I,
        m=mass, 
        dt=delta_time
    )

    """
    Lang = ang_momentum(omega_one, omega_two, omega_three,I)
    Tang = ang_energy(omega_one, omega_two, omega_three,I)
    Llin = lin_momentum(com_x, com_y, com_z, m, dt)
    Tlin = lin_energy(com_x, com_y, com_z, m, dt)
    T = Tlin + Tang
    L = Llin[3] + Lang[3]

    # combine into a single array for energy and momentum
    T = jnp.column_stack((Tang, Tlin, T))
    L = jnp.column_stack((jnp.array(Lang).T, jnp.array(Llin).T, L))

    return T, L

