import jax
import jax.numpy as jnp
from typing import List, Tuple

from mesh import Element, Mesh


class Triangle(Element):
    """ Represents a triangle as 3 vertex indices. """

    def __init__(self, indices: List[int]):
        assert len(indices) == 3
        self.indices = indices

    def reverse_winding(self) -> None:
        """ Reverse the winding order of the triangle. """
        self.flip_orientation()


class TriangleMesh(Mesh):
    """ Represents a triangle mesh as a list of triangles. """

    def __init__(self, triangles: List[Triangle], vertices: jax.Array):
        assert vertices.shape[1] == 3
        assert len(vertices.shape) == 2

        self.elements = triangles
        self.vertices = vertices
