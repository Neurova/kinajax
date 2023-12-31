import jax
from jax import jit
import jax.numpy as jnp





SEG_LENGTH = 16 / 12
SEG_MAX_RADIUS = 4 / 12 / 2
SEG_MASS = 3.5


## SIMULATE 5 MEASUREMENTS
seg_length_measures = jnp.linspace(0,SEG_LENGTH, 5)
seg_rad_measures = jnp.array([SEG_MAX_RADIUS * .65, SEG_MAX_RADIUS * .8, SEG_MAX_RADIUS * .9, SEG_MAX_RADIUS, SEG_MAX_RADIUS * .9])

## GET GRID

X,Y,Z = jnp.meshgrid(
    jnp.linspace(-3, 3, 100),
    jnp.linspace(-3, 3, 100),
    jnp.linspace(0,7,100),
    indexing="ij"
)

@jit
def conical_cylinder_model(X,Y,Z,seg_length, seg_rad):
    """
    Constructs a 3D conical cylinder model of a rigid body from a mesh grid 
    by fitting a polynomial to the radius as a function of cylinder length.

    Parameters
    ----------
    X,Y,Z : array_like
        3D mesh grids of global coordinates
    seg_length : array_like
        n measures along the vertical axis of the cylinder
    seg_rad : array_like
        n radial measures corresponding to the radius at each point along the vertical axis of the cylinder
    poly : int
        the degree of polynomial fit to the radius

    
    Returns
    -------
    seg_array: array_like
        boolean values corresponding to the shape of the conical cylinder inside the mesh grid
    
        
    Raises
    ------
    Value Error
        If grids are not the same shapes
        If shape is not 3D
        If the number of length and radial measurements are not equal
        If the degree of polynomial fit is too high for the number of measurements 


    Example Call
    ------------
    >>> SEG_LENGTH = 16 / 12
    >>> SEG_MAX_RADIUS = 4 / 12 / 2
    >>> SEG_MASS = 3.5

    ## SIMULATE 5 MEASUREMENTS
    >>> seg_length_measures = jnp.linspace(0,SEG_LENGTH, 5)
    >>> seg_rad_measures = jnp.array([SEG_MAX_RADIUS * .65, SEG_MAX_RADIUS * .8, SEG_MAX_RADIUS * .9, SEG_MAX_RADIUS, SEG_MAX_RADIUS * .9])

    >>> seg_array = conical_cylinder_model(X,Y,Z,seg_length_measures, seg_rad_measures)

    """
    n_length_measures = len(seg_length)
    n_rad_mesures = len(seg_rad)

    if X.shape != Y.shape:
        raise ValueError("XYZ Grid shapes must be the same")
    
    if Y.shape != Z.shape:
        raise ValueError("XYZ Grid shapes must be the same")
    
    if X.ndim != 3:
        raise ValueError(f"Must be a 3D cylinder, not {X.ndim} dimensions")

    if n_length_measures != n_rad_mesures:
        raise ValueError("Number of length measurements and radial measurement must be the same")
    
    if n_length_measures < 2:
        raise ValueError(f"Not enough measurements to fit {2} degree polynomial")     
    
    z = Z[0,0,:]
    y = jnp.zeros_like(z)

    fit = jnp.polyfit(seg_length, seg_rad, 2)
    rad = jnp.polyval(fit, z)

    for i in range(n_length_measures):
        if i == n_length_measures-1:
            y = jnp.where(z >= seg_length[i], 0, y)
        else:
            condition = (z >= seg_length[i]) & (z < seg_length[i+1])
            # updated_values = rad[condition]
            y = jnp.where(condition, rad, y)

    body_array = (jnp.hypot(X, Y) < y[None, None, :])

    return body_array


@jit
def elliptical_cone_model(X, Y, Z, seg_length, seg_rad, minor_rad):
    """
    Constructs a 3D elliptical cone model of a rigid body from a mesh grid 
    by fitting a polynomial to the radius as a function of cone length and setting 
    the minor radius to a constant value.

    Parameters
    ----------
    X,Y,Z : array_like
        3D mesh grids of global coordinates
    seg_length : array_like
        n measures along the vertical axis of the cylinder
    seg_rad : array_like
        n radial measures corresponding to the radius at each point along the vertical axis of the cylinder
    poly : int
        the degree of polynomial fit to the radius

    
    Returns
    -------
    seg_array: array_like
        boolean values corresponding to the shape of the conical cylinder inside the mesh grid
    
        
    Raises
    ------
    Value Error
        If grids are not the same shapes
        If shape is not 3D
        If the number of length and radial measurements are not equal
        If the degree of polynomial fit is too high for the number of measurements 


    Example Call
    ------------
    >>> SEG_LENGTH = 16 / 12
    >>> SEG_MAX_RADIUS = 4 / 12 / 2
    >>> SEG_MASS = 3.5

    ## SIMULATE 5 MEASUREMENTS
    >>> seg_length_measures = jnp.linspace(0,SEG_LENGTH, 5)
    >>> seg_rad_measures = jnp.array([SEG_MAX_RADIUS * .65, SEG_MAX_RADIUS * .8, SEG_MAX_RADIUS * .9, SEG_MAX_RADIUS, SEG_MAX_RADIUS * .9])

    >>> seg_array = elliptical_cone_model(X,Y,Z,seg_length_measures, seg_rad_measures)

    """

    n_length_measures = len(seg_length)
    n_rad_mesures = len(seg_rad)

    if X.shape != Y.shape:
        raise ValueError("XYZ Grid shapes must be the same")
    
    if Y.shape != Z.shape:
        raise ValueError("XYZ Grid shapes must be the same")
    
    if X.ndim != 3:
        raise ValueError("Must be a 3D cylinder, not {X.ndims} dimensions")

    if n_length_measures != n_rad_mesures:
        raise ValueError("Number of length measurements and radial measurement must be the same")
    


    # Polynomial fit to major radius
    fit = jnp.polyfit(seg_length, seg_rad, 2)
    max_z = jnp.max(seg_length)

    # Calculate the elliptical cylinder model
    r_major = jnp.where(Z > max_z, 0, jnp.polyval(fit, Z))
    r_minor = jnp.where(Z > max_z, 0, minor_rad)
    body_array = (X**2/r_major**2 + Y**2/r_minor**2) <= 1

    return body_array

@jit
def sampled_volume(X,Y,Z,body_array):
        """
        Computes the integral volume of a 3D rigid body from sampled data.
            If the the data is sampled at constant intervals then this is also equivelant to

            dx = jnp.unique(jnp.diff(X[:,0,0]))[0]
            dy = jnp.unique(jnp.diff(Y[0,:,0]))[0]
            dz = jnp.unique(jnp.diff(Z[0,0,:]))[0]

            V = jnp.sum(jnp.sum(jnp.sum(body_array, axis=0)*dx,axis=0)*dy)*dz

            OR

            V = jnp.trapz(np.trapz(np.trapz(body_array, dx=dx, axis=2), dx=dy, axis=1), dx=dz, axis=0)

        Parameters
        ----------
        X,Y,Z : array_like
            3D mesh grids of global coordinate system
        body_array : array_like
            boolean or float values corresponding to the shape of the conical cylinder inside the mesh grid

                
        Returns
        -------
            V : float
                volume of 3D rigid body

                
        Raises
        ------
        ValueError
            If grids are not the same shapes
            If shape is not 3D
            If body_array has not been formed yet

        Example Call
        ------------
        >>> SEG_LENGTH = 16 / 12
        >>> SEG_MAX_RADIUS = 4 / 12 / 2
        >>> SEG_MASS = 3.5


        ## SIMULATE 5 MEASUREMENTS
        >>> seg_length_measures = np.linspace(0,SEG_LENGTH, 5)
        >>> seg_rad_measures = np.array([SEG_MAX_RADIUS * .65, SEG_MAX_RADIUS * .8, SEG_MAX_RADIUS * .9, SEG_MAX_RADIUS, SEG_MAX_RADIUS * .9])

        >>> body_array = conical_cylinder_model(seg_length_measures, seg_rad_measures, poly=2)

        >>> V = sampled_volume(X,Y,Z,body_array)

        """

        if X.shape != Y.shape:
            raise ValueError("XYZ Grid shapes must be the same")
        
        if Y.shape != Z.shape:
            raise ValueError("XYZ Grid shapes must be the same")
        
        if body_array is None:
            raise ValueError("Must construct the rigid body array first through one of the rigid body models available")
        
        if body_array.ndim != 3:
            raise ValueError("Must be a 3D rigid body, not {body_array.ndims} dimensions")
        
        body_array = body_array.astype(float)

        # axis default 0
        # could also be 2, 1, 0
        V = jnp.trapz(jnp.trapz(jnp.trapz(body_array,x=X[:,0,0]), x=Y[0,:,0]), x=Z[0,0,:])

        return V

@jit
def com_symmetrical_top(X,Y,Z,body_array,V):
    """
    Computes the center of mass of a symmetrical top.  Meaning the X and Y CoM measures are assumed 0.

    Parameters
    ----------
    X,Y,Z : array_like
        3D mesh grids of global coordinate system
    body_array : array_like
        boolean or float values corresponding to the shape of the conical cylinder inside the mesh grid
    V : float
        volume of rigid body

    Returns
    --------
    (com_x, com_y, com_z) : tuple
        floats that correspond to the center of mass of the rigid body along the x,y, and z axis

    Raises
    ---- --
    If grid shapes are not the same

    Example Call
    ------------
    >>> SEG_LENGTH = 16 / 12
    >>> SEG_MAX_RADIUS = 4 / 12 / 2
    >>> SEG_MASS = 3.5


    ## SIMULATE 5 MEASUREMENTS
    >>> seg_length_measures = jnp.linspace(0,SEG_LENGTH, 5)
    >>> seg_rad_measures = jnp.array([SEG_MAX_RADIUS * .65, SEG_MAX_RADIUS * .8, SEG_MAX_RADIUS * .9, SEG_MAX_RADIUS, SEG_MAX_RADIUS * .9])

    >>> body_array = conical_cylinder_model(X,Y,Z, seg_length_measures, seg_rad_measures, poly=22)
    >>> V = sampled_volume(X,Y,Z,body_array)
    >>> com_x, com_y, com_z = com_symmetrical_top(X,Y,Z,body_array,V)
    """

    if X.shape != Y.shape:
        raise ValueError("XYZ Grid shapes must be the same")
    
    if Y.shape != Z.shape:
        raise ValueError("XYZ Grid shapes must be the same")
    
    body_array = body_array.astype(float)

    com_z = (
        jnp.trapz(
            jnp.trapz(
                jnp.trapz(body_array * Z, x=X[:, 0, 0], axis=0),
                    x=Y[0, :, 0], axis=0,
            ), x=Z[0, 0, :], axis=0,
        )
        / V
    )
    com_x = com_y = 0.0

    return (com_x, com_y, com_z)

@jit
def inertia_tensor(X,Y,Z,body_array,p):
        """
        Computes the diagonals of the inertia tensor which correspond to the mass moments of inertia.  Asserts off diagonals are zero.

        Also equivalent to the discrete summation:

            dx = jnp.unique(jnp.diff(X[:,0,0]))[0]
            dy = jnp.unique(jnp.diff(Y[0,:,0]))[0]
            dz = jnp.unique(jnp.diff(Z[0,0,:]))[0]
            r_squared = X**2 + Y**2 + Z**2

            I[0, 0] = jnp.sum(p*(r_squared - X**2) * body_array * dx * dy * dz)
            I[1, 1] = jnp.sum(p*(r_squared - Y**2) * body_array * dx * dy * dz)
            I[2, 2] = jnp.sum(p*(r_squared - Z**2) * body_array * dx * dy * dz)

            I[0, 1] = I[1, 0] = -jnp.sum(p*(X*Y) * body_array * dx * dy * dz)
            I[0, 2] = I[2, 0] = -jnp.sum(p*(X*Z) * body_array * dx * dy * dz)
            I[1, 2] = I[2, 1] = -jnp.sum(p*(Y*Z) * body_array * dx * dy * dz)


        If interested in the off diagonals as an integral then it can also be written completely as:
            r_squared = X**2 + Y**2 + Z**2
            I_mat_factors = np.array(
                [[X * X, X * Y, X * Z], [Y * X, Y * Y, Y * Z], [Z * X, Z * Y, Z * Z]]
            )
            I = np.empty((3, 3))
            for i in range(3):
                for j in range(3):
                    if i == j:
                        coef = r_squared - I_mat_factors[i, j]
                    else:
                        coef = -I_mat_factors[i, j]
                    I[i, j] = jnp.trapz(jnp.trapz(jnp.trapz(p * body_array.astype(float) * coef, x=X[:, 0, 0]),x=Y[0, :, 0]), x=Z[0, 0, :])
        
        Parameters
        ----------
        X,Y,Z : array_like
            3D mesh grids of global coordinate system
        body_array : array_like
            boolean or float values corresponding to the shape of the conical cylinder inside the mesh grid
        p : float
            density of the rigid body

        Returns
        -------
            I: array_like
                a 3x3 array with the diagonals corresponding the the mass moments of inertia on the x,y,z axis

                
        Raises
        ------
        ValueError
            If grids are not the same shapes
            If shape is not 3D

        
        Example Call
        ------------
        >>> SEG_LENGTH = 16 / 12
        >>> SEG_MAX_RADIUS = 4 / 12 / 2
        >>> SEG_MASS = 3.5


        ## SIMULATE 5 MEASUREMENTS
        >>> seg_length_measures = jnp.linspace(0,SEG_LENGTH, 5)
        >>> seg_rad_measures = jnp.array([SEG_MAX_RADIUS * .65, SEG_MAX_RADIUS * .8, SEG_MAX_RADIUS * .9, SEG_MAX_RADIUS, SEG_MAX_RADIUS * .9])

        >>> body_array = conical_cylinder_model(seg_length_measures, seg_rad_measures, poly=2)
        >>> V = sampled_volume(X,Y,Z,body_array)
        >>> p = SEG_MASS / V

        >>> I = inertia_tensor(X,Y,Z,body_array,p)
        """

        if X.shape != Y.shape:
            raise ValueError("XYZ Grid shapes must be the same")
        
        if Y.shape != Z.shape:
            raise ValueError("XYZ Grid shapes must be the same")
        
        if body_array.ndim != 3:
            raise ValueError("Must be a 3D rigid body, not {body_array.ndims} dimensions")

        
        I = jnp.eye(3)
        r_squared = X**2 + Y**2 + Z**2
        body_array = body_array.astype(float)

        # axis default 0
        # could also be 2, 1, 0
        I = I.at[0,0].set(jnp.trapz(jnp.trapz(jnp.trapz(p * (r_squared - X**2) * body_array, x=X[:,0,0]), x=Y[0,:,0]), x=Z[0,0,:]))
        I = I.at[1,1].set(jnp.trapz(jnp.trapz(jnp.trapz(p * (r_squared - Y**2) * body_array, x=X[:,0,0]), x=Y[0,:,0]), x=Z[0,0,:]))
        I = I.at[2,2].set(jnp.trapz(jnp.trapz(jnp.trapz(p * (r_squared - Z**2) * body_array, x=X[:,0,0]), x=Y[0,:,0]), x=Z[0,0,:]))


        return I



####################
## EXAMPLE CALLS
####################


body_array = conical_cylinder_model(X,Y,Z,seg_length_measures, seg_rad_measures)
#elliptical_cone_model(X,Y,Z,seg_length_measures, seg_rad_measures,.4)
V = sampled_volume(X,Y,Z,body_array)
com_x, com_y, com_z = com_symmetrical_top(X,Y,Z,body_array,V)
p = SEG_MASS / V
I = inertia_tensor(X,Y,Z,body_array,p)
