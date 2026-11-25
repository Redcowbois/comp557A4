import taichi as ti
import taichi.math as tm

@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3

@ti.func
def getRayDistance(ray: Ray, point: tm.vec3) -> float:
    return tm.length(point - ray.origin)

@ti.func
def getRayPoint(ray: Ray, t: float) -> tm.vec3:
    return ray.origin + ray.direction * t

@ti.func
def changeRayFrame(ray: Ray, M: tm.mat4) -> Ray:
    # Objective 4: Ray and Geometry Transformations
    orig_h = tm.vec4(ray.origin.x, ray.origin.y, ray.origin.z, 1.0)
    dir_h = tm.vec4(ray.direction.x, ray.direction.y, ray.direction.z, 0.0)

    new_orig = M @ orig_h
    new_dir = M @ dir_h

    new_orig_3 = tm.vec3(new_orig.x, new_orig.y, new_orig.z)
    if new_orig.w != 1.0 and new_orig.w != 0.0:
        new_orig_3 /= new_orig.w
    new_dir_3 = tm.vec3(new_dir.x, new_dir.y, new_dir.z)

    return Ray(new_orig_3, new_dir_3)


@ti.dataclass
class Material:
    texture_id: int     # texture ID, -1 if no texture
    diffuse: tm.vec3    # kd diffuse coefficient, modulates texture if texture present
    specular: tm.vec3    # ks specular coefficient
    shininess: float    # specular exponent

@ti.dataclass
class Light:
    ltype: int       # type is either 0 for "directional" or 1 for "point"
    colour: tm.vec3   # colour and intensity of the light
    vector: tm.vec3    # position, or normalized direction towards light, depending on the light type
    attenuation: tm.vec3   # attenuation coeffs [quadratic, linear, constant] for point lights

@ti.dataclass 
class Intersection: # All fields will be set to zero on creation, otherwise specified in this order on construction
    is_hit: bool
    t: float
    normal: tm.vec3
    position: tm.vec3
    mat: int
    texture_coords: tm.vec3  # could be 2D (first 2 coords) or 3d texture coordinates 

@ti.func
def changeIntersectFrame(intersect: Intersection, M: tm.mat4, M_inv: tm.mat4) -> Intersection:
    # Objective 4: Ray and Geometry Transformations
    pos_h = tm.vec4(intersect.position.x, intersect.position.y, intersect.position.z, 1.0)
    new_pos = M @ pos_h

    norm_h = tm.vec4(intersect.normal.x, intersect.normal.y, intersect.normal.z, 0.0)
    new_norm = M_inv.transpose() @ norm_h

    out = intersect
    out.position = tm.vec3(new_pos.x, new_pos.y, new_pos.z)
    if new_pos.w != 1.0 and new_pos.w != 0.0:
        out.position /= new_pos.w
    out.normal = tm.normalize(tm.vec3(new_norm.x, new_norm.y, new_norm.z))

    return out