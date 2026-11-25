
from helperclasses import Ray
import taichi as ti
from pyglm import glm
import taichi.math as tm

@ti.data_oriented
class Camera:
	def __init__(self, width, height, eye_position:glm.vec3, lookat:glm.vec3, up:glm.vec3, fovy, jitter:bool=False, samples:int=1) -> None:
		'''	Initialize the camera with given parameters.
		Args:
			width (int): width of the image in pixels
			height (int): height of the image in pixels
			eye_position (glm.vec3): position of the camera in world space
			lookat (glm.vec3): point the camera is looking at
			up (glm.vec3): up direction for the camera
			fovy (float): vertical field of view in degrees
			jitter (bool): whether to apply jittering within the pixel for anti-aliasing
			samples (int): number of samples per pixel for anti-aliasing
		Returns:
			None

			Note that while glm types are provided to construct the class, the 
			member variables must be of taich tm.vec3 types so that they used 
			in the create_ray Taichi function.  
		'''
		self.width = width
		self.height = height		
		self.distance_to_plane = 1.0
		self.jitter = jitter
		self.samples = samples

		# Objective 1: Compute camera frame basis vectors, and top bottom left right for ray generation
		# w points opposite to the gaze direction (away from scene)
		w_glm = glm.normalize(eye_position - lookat)
		# u is the right vector (up x w)
		u_glm = glm.normalize(glm.cross(up, w_glm))
		# v is the true orthogonal up vector (w x u)
		v_glm = glm.cross(w_glm, u_glm)

		# Compute view plane boundaries at distance_to_plane
		fovy_rad = glm.radians(fovy)
		self.top = self.distance_to_plane * glm.tan(fovy_rad / 2.0)
		self.bottom = -self.top
		aspect_ratio = float(width) / float(height)
		self.right = self.top * aspect_ratio
		self.left = -self.right

		# Store basis vectors as Taichi tm.vec3 types
		self.u = tm.vec3(u_glm.x, u_glm.y, u_glm.z)
		self.v = tm.vec3(v_glm.x, v_glm.y, v_glm.z)
		self.w = tm.vec3(w_glm.x, w_glm.y, w_glm.z)
		self.eye_position = tm.vec3(eye_position.x, eye_position.y, eye_position.z)



	@ti.func
	def create_ray(self, x, y) -> Ray:
		''' Create a ray going through pixel (x,y) in image space 
		Args:
			x (int): pixel x coordinate
			y (int): pixel y coordinate
		Returns:
			Ray: generated ray from camera through pixel

			If jitter is True, the ray should not go through the exact center of the
			pixel, but instead be through a random point (uniform) inthe pixel area.
		'''

		
		# Objective 1: Generate ray from camera through pixel (x, y) with optional jittering
		offset_x = 0.5
		offset_y = 0.5
		if ti.static(self.jitter):
			offset_x = ti.random()
			offset_y = ti.random()

		u_val = self.left + (self.right - self.left) * (x + offset_x) / self.width
		v_val = self.bottom + (self.top - self.bottom) * (y + offset_y) / self.height

		# Direction vector: s - e = u * u_basis + v * v_basis - d * w_basis
		direction = tm.normalize(u_val * self.u + v_val * self.v - self.distance_to_plane * self.w)

		return Ray(self.eye_position, direction)
