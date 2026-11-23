# Assignment 4 Codebase Overview: Taichi Ray Tracer

This repository contains the starter code and assets for **COMP 557 (Computer Graphics) Assignment 4: Ray Tracing**. The project implements a GPU/CPU-accelerated ray tracer using the **Taichi** programming language (`taichi`) alongside `pyglm` and `numpy`.

---

## Architecture & Code Structure

### 1. `main.py`
The CLI entry point for running the ray tracer.
- **CLI Arguments**: Supports input scene path (`-i`/`--infile`), output directory (`-o`/`--outdir`), display GUI window (`-s`/`--show`), resolution scaling factor (`-f`/`--factor`), and Taichi compute backend (`-ti`/`--taichi`: `cpu`, `cuda`, `vulkan`, `metal`).
- **Render Loop**: Initializes Taichi, instantiates a `Scene`, invokes `parser.load_scene(...)`, and renders pixel samples iteratively (with support for progressive GUI updates or headless batch rendering).
- **Image Export**: Converts the resulting Taichi image field to NumPy, flips/formats coordinate conventions, and writes a `.png` via PIL.

### 2. `camera.py`
Defines the `Camera` class decorated with `@ti.data_oriented`.
- **`__init__`**: Stores viewport dimensions, field of view (`fovy`), eye position, look-at target, and up vector.
  - *Objective 1*: Compute camera basis vectors $(\mathbf{u}, \mathbf{v}, \mathbf{w})$ and view-plane boundaries (`top`, `bottom`, `left`, `right`).
- **`create_ray(x, y)`**: Generates a world-space `Ray` through pixel coordinates $(x, y)$, incorporating stratified/uniform sub-pixel jitter when `jitter` is enabled.

### 3. `helperclasses.py`
Core data structures and vector transformation helpers decorated with `@ti.dataclass` / `@ti.func`:
- **`Ray`**: Holds `origin` and `direction` (`tm.vec3`).
- **`Intersection`**: Contains hit record attributes (`is_hit`, distance `t`, surface `normal`, hit `position`, material index `mat`, and `texture_coords`).
- **`Material`**: Holds diffuse coefficient `diffuse`, specular coefficient `specular`, shininess exponent, and optional `texture_id`.
- **`Light`**: Stores light type (`0` for directional, `1` for point), light `colour`, direction/position `vector`, and distance `attenuation` factors.
- **`changeRayFrame` & `changeIntersectFrame`**:
  - *Objective 4*: Transform rays into object local coordinates using $M^{-1}$ and map hit results (points and surface normals) back into world space using $M$ and $(M^{-1})^T$.

### 4. `geometry.py`
Intersection tests against geometric primitives:
- **`Sphere` / `intersectSphere`**: *Objective 2* — Algebraic ray-sphere quadratic intersection.
- **`Plane` / `intersectPlane`**: *Objective 5* — Ray-plane intersection with support for checkerboard texturing / material indexing.
- **`AABox` / `intersectAABox`**: *Objective 7* — Ray-axis-aligned box intersection via the slab method.
- **`Mesh` / `intersectMesh`**: *Objective 8* — Ray-triangle mesh intersection testing against global vertex/face buffers (e.g., using barycentric coordinates or Möller–Trumbore).

### 5. `scene.py`
Central scene management and rendering kernel:
- **`Scene`**: Allocates Taichi fields for spheres, planes, boxes, meshes, lights, materials, textures, and image output.
- **`load(...)`**: Populates Taichi fields from parsed scene data.
- **`render()`**: Main Taichi `@ti.kernel` parallelizing ray generation and shading across pixels $(x, y)$ in `ti.ndrange(width, height)`.
- **`intersect_scene(ray, t_min, t_max)`**: Iterates over all scene primitives and tracks the closest intersection $t_{\min} \le t \le t_{\max}$.
- **`compute_shading(intersect, ray)`**:
  - Ambient shading.
  - *Objective 3*: Blinn-Phong / diffuse & specular shading calculations across point and directional lights.
  - *Objective 6*: Shadow ray occlusion testing (`shadow_epsilon`).
- **`getTextureColour(...)`**: Texture coordinate lookup with UV wrapping.

### 6. `parser.py`
JSON scene file loader using `json5`.
- Parses camera settings, ambient illumination, light sources, texture maps, materials, meshes (`trimesh`), and hierarchical scene node graphs (flattening transformation matrices down to leaves).

### 7. `run_all.py`
Convenience batch script to discover and render all `.json` scenes under `scenes/`.

---

## Directory Structure & Assets

- **`scenes/`**: Test scenes exercising various objectives:
  - `Sphere.json`, `TwoSpheresPlane.json`, `Plane.json`, `Plane2.json`: Basic primitives and plane tests.
  - `AACheckerPlane.json`, `NoAACheckerPlane.json`: Super-sampling and anti-aliasing jitter verification.
  - `Ellipse.json`: Object transformations ($M$ and $M^{-1}$).
  - `BoxRGBLights.json`, `BoxStacks.json`: Box primitive intersections and shadow interactions.
  - `BunnyMesh.json`, `TorusMesh.json`, `TwoMeshes.json`, `TwoMeshesSimple.json`: Triangle mesh intersections.
  - `3HHB.json`, `3HHBsmall.json`: Complex hierarchical scenes.
- **`meshes/`**: `.obj` 3D models (`bunny.obj`, `cube.obj`, `tetrahedron.obj`, `torus.obj`).
- **`textures/`**: Test textures (`black-white-2x2.png`, `uv_test.png`, etc.).
- **`out/`**: Destination directory for output rendered `.png` images.
- **`comp557F26A4.pdf`**: Lecture slides covering radiometry, Monte Carlo integration, and ray tracing theory.

---

## Assignment Objectives Roadmap

1. **Objective 1**: Camera setup, basis vector derivation, viewing ray generation, and jittered super-sampling (`camera.py`).
2. **Objective 2**: Ray-sphere intersection (`geometry.py`).
3. **Objective 3**: Diffuse and specular Blinn-Phong lighting (`scene.py`).
4. **Objective 4**: Affine transformations for rays and intersections (`helperclasses.py`).
5. **Objective 5**: Ray-plane intersection and checkerboard texturing (`geometry.py`).
6. **Objective 6**: Shadow rays and light attenuation (`scene.py`).
7. **Objective 7**: Ray-box intersection using the slab method (`geometry.py`).
8. **Objective 8**: Ray-triangle mesh intersection (`geometry.py`).
