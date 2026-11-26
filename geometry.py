from helperclasses import Ray, Intersection, Material, changeRayFrame, getRayPoint, changeIntersectFrame

import taichi as ti
import taichi.math as tm

EPSILON = 10 ** (-5)

@ti.dataclass
class Sphere:
    material_id: int
    radius: float
    M: tm.mat4
    M_inv: tm.mat4

@ti.func
def intersectSphere(sphere: Sphere, ray: Ray, t_min: float, t_max: float) -> Intersection:
    ''' Ray-sphere intersection  
    Args:
        sphere (Sphere): sphere to intersect with
        ray (Ray): ray in world space
        t_min (float): minimum t value for valid intersection
        t_max (float): maximum t value for valid intersection
    Returns:
        Intersection: intersection data (is_hit, t, normal, point, material)
        
        Note, nominally the intersection will be computed with the sphere
        at the origin with the specified radius.  For a sphere away from 
        the origin, the M and M_inv members of the Sphere class must be used to 
        transform the ray into the sphere's local frame for intersection, and
        likewise used to transform the resulting intersection data back to world space
        (TODO: See Objective ?).
    '''

    hit = Intersection() # default is no intersection (is_hit = False)

    # Objective 2: Implement ray-sphere intersection
    local_ray = changeRayFrame(ray, sphere.M_inv)

    p = local_ray.origin
    d = local_ray.direction
    R = sphere.radius

    a = tm.dot(d, d)
    b_half = tm.dot(p, d)
    c = tm.dot(p, p) - R * R

    discriminant = b_half * b_half - a * c

    if discriminant >= 0.0:
        sqrt_disc = tm.sqrt(discriminant)
        t1 = (-b_half - sqrt_disc) / a
        t2 = (-b_half + sqrt_disc) / a

        t_hit = 0.0
        hit_found = False
        if t1 >= t_min and t1 <= t_max:
            t_hit = t1
            hit_found = True
        elif t2 >= t_min and t2 <= t_max:
            t_hit = t2
            hit_found = True

        if hit_found:
            local_pos = getRayPoint(local_ray, t_hit)
            local_normal = tm.normalize(local_pos) # outward normal
            hit.is_hit = True
            hit.t = t_hit
            hit.position = local_pos
            hit.normal = local_normal
            hit.mat = sphere.material_id
            hit = changeIntersectFrame(hit, sphere.M, sphere.M_inv)

    return hit



@ti.dataclass
class Plane:
    material_id: int    
    normal: tm.vec3         # only plane normal (goes through origin in local space)
    M: tm.mat4              # transformation matrix to world space  
    M_inv: tm.mat4          # inverse transformation matrix to local space  

@ti.func
def intersectPlane(plane: Plane, ray: Ray, t_min: float, t_max: float) -> Intersection:
    ''' Ray-plane intersection
    Args:
        plane (Plane): plane to intersect with
        ray (Ray): ray in world space
        t_min (float): minimum t value for valid intersection
        t_max (float): maximum t value for valid intersection
    Returns:
        Intersection: intersection data (is_hit, t, normal, point, material)
        
        Note, nominally the intersection will be computed with the plane
        at the origin with the specified normal.  For a plane away from 
        the origin, the M and M_inv members of the Plane class must be used to 
        transform the ray into the plane's local frame for intersection, and
        likewise used to transform the resulting intersection data back to world space.

        Note, the plane can have two materials (when two_materials is true) which
        are applied in a checkerboard pattern in the plane's local XZ plane.
    '''

    hit = Intersection() # default is no intersection (is_hit = False)

    # Objective 5: Implement ray-plane intersection
    local_ray = changeRayFrame(ray, plane.M_inv)

    denom = tm.dot(local_ray.direction, plane.normal)
    if ti.abs(denom) > 1e-6:
        t = -tm.dot(local_ray.origin, plane.normal) / denom
        if t >= t_min and t <= t_max:
            local_pos = getRayPoint(local_ray, t)
            hit.is_hit = True
            hit.t = t
            hit.position = local_pos
            hit.normal = plane.normal
            hit.mat = plane.material_id
            hit.texture_coords = tm.vec3(local_pos.x, local_pos.z, 0.0)
            hit = changeIntersectFrame(hit, plane.M, plane.M_inv)

    return hit


@ti.dataclass
class AABox:
    material_id: int
    minpos: tm.vec3     # lower left corner
    maxpos: tm.vec3     # upper right corner
    M: tm.mat4
    M_inv: tm.mat4

@ti.func
def intersectAABox(aabox: AABox, ray: Ray, t_min: float, t_max: float) -> Intersection:
    hit = Intersection() # default is no intersection (is_hit = False)

    # Objective 7: Ray-AABox intersection using the slab method
    local_ray = changeRayFrame(ray, aabox.M_inv)
    p = local_ray.origin
    d = local_ray.direction

    # Compute t intervals for x slab
    tx_min = (aabox.minpos.x - p.x) / d.x
    tx_max = (aabox.maxpos.x - p.x) / d.x
    tx_entry = ti.min(tx_min, tx_max)
    tx_exit = ti.max(tx_min, tx_max)

    # Compute t intervals for y slab
    ty_min = (aabox.minpos.y - p.y) / d.y
    ty_max = (aabox.maxpos.y - p.y) / d.y
    ty_entry = ti.min(ty_min, ty_max)
    ty_exit = ti.max(ty_min, ty_max)

    # Compute t intervals for z slab
    tz_min = (aabox.minpos.z - p.z) / d.z
    tz_max = (aabox.maxpos.z - p.z) / d.z
    tz_entry = ti.min(tz_min, tz_max)
    tz_exit = ti.max(tz_min, tz_max)

    t_entry = ti.max(ti.max(tx_entry, ty_entry), tz_entry)
    t_exit = ti.min(ti.min(tx_exit, ty_exit), tz_exit)

    if t_exit >= t_entry:
        t_hit = -1.0
        hit_normal = tm.vec3(0.0)
        if t_entry >= t_min and t_entry <= t_max:
            t_hit = t_entry
            if t_entry == tx_entry:
                hit_normal = tm.vec3(-1.0 if tx_entry == tx_min else 1.0, 0.0, 0.0)
            elif t_entry == ty_entry:
                hit_normal = tm.vec3(0.0, -1.0 if ty_entry == ty_min else 1.0, 0.0)
            else:
                hit_normal = tm.vec3(0.0, 0.0, -1.0 if tz_entry == tz_min else 1.0)
        elif t_exit >= t_min and t_exit <= t_max:
            t_hit = t_exit
            if t_exit == tx_exit:
                hit_normal = tm.vec3(1.0 if tx_exit == tx_max else -1.0, 0.0, 0.0)
            elif t_exit == ty_exit:
                hit_normal = tm.vec3(0.0, 1.0 if ty_exit == ty_max else -1.0, 0.0)
            else:
                hit_normal = tm.vec3(0.0, 0.0, 1.0 if tz_exit == tz_max else -1.0)

        if t_hit >= 0.0:
            hit.is_hit = True
            hit.t = t_hit
            hit.position = getRayPoint(local_ray, t_hit)
            hit.normal = hit_normal
            hit.mat = aabox.material_id
            hit = changeIntersectFrame(hit, aabox.M, aabox.M_inv)

    return hit


@ti.dataclass
class Mesh:
    material_id: int
    faces_ids_start: ti.i32  # index of the first face in the global face array
    faces_ids_count: ti.i32  # number of faces in this mesh
    M: tm.mat4
    M_inv: tm.mat4

@ti.func
def intersectMesh(mesh: Mesh,                  # data for this mesh (start face and number of faces, etc.)
                  meshes_verts: ti.template(), # all vertices (for all meshes)
                  meshes_faces: ti.template(), # all faces (for all meshes)
                  ray: Ray,
                  t_min: float,
                  t_max: float
) -> Intersection:
    
    out_intersect = Intersection() # default is no intersection (is_hit = False)

    # TODO: Objective 8: Implement ray-mesh intersection


    return out_intersect
