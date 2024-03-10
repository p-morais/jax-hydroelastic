import jax
import jax.numpy as jnp
from typing import List, Tuple

from mesh import Element, Mesh


class Tetrahedron(Element):
    """ Represents a tetrahedron as 4 vertex indices. """

    def __init__(self, indices: List[int]):
        assert len(indices) == 4
        self.indices = indices


class VolumeMesh(Mesh):
    """ Represents a volume mesh as a list of tetrahedra. """

    def __init__(self, tetrahedra: List[Tetrahedron], vertices: jax.Array):
        assert vertices.shape[1] == 3
        assert len(vertices.shape) == 2

        self.elements = tetrahedra
        self.vertices = vertices


tet = Tetrahedron([1, 2, 3, 4])
verts = jnp.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

mesh = VolumeMesh([tet], verts)
mesh.print()

matrix = jnp.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

mesh.transform(matrix, jnp.zeros(3, ))
mesh.print()
