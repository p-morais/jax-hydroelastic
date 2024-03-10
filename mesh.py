from abc import ABC, abstractmethod
from typing import List, Tuple
import jax
import jax.numpy as jnp


class Element(ABC):

    def flip_orientation(self) -> None:
        """ Flip the orientation of the tetrahedron. """

        # You can flip the orientation of a tetrahedron or winding order of a triangle
        # by swapping the first two indices.
        self.indices[0], self.indices[1] = self.indices[1], self.indices[0]

    def print(self) -> None:
        print(self.indices)


class Mesh(ABC):

    def num_elements(self) -> int:
        return len(self.elements)

    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def get_element(self, index: int) -> Element:
        return self.elements[index]

    def get_vertex(self, index: int) -> jax.Array:
        return self.vertices[index]

    def transform(self, rotation: jax.Array, translation: jax.Array) -> None:
        assert rotation.shape == (3, 3)
        assert translation.shape == (3, )
        self.vertices = jax.lax.batch_matmul(self.vertices, rotation) + translation

    def print(self) -> None:
        print("elements: ")
        for element in self.elements:
            element.print()
        print("vertices: ")
        print(self.vertices)
