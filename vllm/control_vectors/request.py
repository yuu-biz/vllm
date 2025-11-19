# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import msgspec


class ControlVectorRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    Request for a ControlVector adapter.

    Note that this class should be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized ControlVector adapters.

    control_vector_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """

    control_vector_name: str
    control_vector_id: int
    control_vector_path: str = ""
    scale: float = 1.0
    base_model_name: str | None = None

    @property
    def adapter_id(self):
        return self.control_vector_id

    @property
    def name(self):
        return self.control_vector_name

    @property
    def path(self):
        return self.control_vector_path

    @property
    def scale_factor(self):
        return self.scale

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare ControlVectorRequest
        instances based on control_vector_name. This allows for identification
        and comparison lora adapter across engines.
        """
        return (
            isinstance(value, self.__class__)
            and self.control_vector_name == value.control_vector_name
        )

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash ControlVectorRequest instances
        based on control_vector_name. This ensures that ControlVectorRequest
        instancescan be used in hash-based collections such as sets and
        dictionaries,identified by their names across engines.
        """
        return hash(self.control_vector_name)
