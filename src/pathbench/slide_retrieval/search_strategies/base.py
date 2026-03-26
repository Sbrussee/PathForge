from __future__ import annotations

from typing import Any, Literal

from pathbench.slide_retrieval.hyperparams import (
    collect_hyperparams,
    resolve_hyperparam,
)
from pathbench.slide_retrieval.search_strategies.types import (
    SearchDatabaseItem,
    SearchHit,
    SearchResult,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


RepresentationKind = Literal["single_vector", "multi_vector"]


class BaseSearchStrategy:
    """Base class for search strategies."""

    name: str = ""
    supported_representation_kinds: frozenset[RepresentationKind] = frozenset()

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.params = params or {}
        self.extra = kwargs or {}
        self.search_database: list[SearchDatabaseItem] = []
        self._bind_hyperparams()

    @classmethod
    def hyperparam_spec(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter schema for this strategy."""
        return {
            name: declaration.to_spec()
            for name, declaration in collect_hyperparams(cls).items()
        }

    def _get_hp(self, key: str) -> Any:
        """Resolve one hyperparameter value."""
        declarations = collect_hyperparams(type(self))
        if key not in declarations:
            raise KeyError(
                f"Hyperparam '{key}' is not declared on strategy '{type(self).__name__}'."
            )
        return resolve_hyperparam(
            name=key,
            declaration=declarations[key],
            params=self.params,
        )

    def hyperparam_values(self) -> dict[str, Any]:
        """Return the effective hyperparameter values for this instance."""
        return {
            name: getattr(self, name)
            for name in collect_hyperparams(type(self))
        }

    def _bind_hyperparams(self) -> None:
        """Resolve and bind all declared hyperparameters to the instance."""
        for name in collect_hyperparams(type(self)):
            setattr(self, name, self._get_hp(name))

    def supports_representation_kind(self, representation_kind: str) -> bool:
        """Return whether this strategy supports the provided representation kind."""
        return representation_kind in self.supported_representation_kinds

    def validate_representation_kind(self, representation_kind: str) -> None:
        """Validate that this strategy supports the provided representation kind."""
        if not self.supported_representation_kinds:
            raise ValueError(
                f"Search strategy '{self.name}' does not define "
                f"supported_representation_kinds."
            )

        if representation_kind not in self.supported_representation_kinds:
            raise ValueError(
                f"Search strategy '{self.name}' does not support representation kind "
                f"'{representation_kind}'. Supported kinds: "
                f"{sorted(self.supported_representation_kinds)}"
            )

    def build_database(
        self,
        database_representations: list[RetrievalRepresentation],
    ) -> None:
        """Build the searchable database from retrieval representations."""
        self._validate_representations(database_representations)

        self.search_database = [
            self.build_database_item(representation)
            for representation in database_representations
        ]

        self.build_index()

    def build_database_item(
        self,
        representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        """Convert one retrieval representation into a searchable database item."""
        return SearchDatabaseItem(
            item_id=representation.sample_id,
            search_type=representation.representation_type,
            data=representation.data,
            metadata=representation.metadata,
        )

    def prepare_query(
        self,
        query_representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        """Convert one query retrieval representation into a searchable query item."""
        self._validate_representations([query_representation])

        return SearchDatabaseItem(
            item_id=query_representation.sample_id,
            search_type=query_representation.representation_type,
            data=query_representation.data,
            metadata=query_representation.metadata,
        )

    def build_index(self) -> None:
        """Optional hook to prepare the searchable database."""
        return None

    def filter_database_by_patient(
        self,
        query_item: SearchDatabaseItem,
        database_items: list[SearchDatabaseItem] | None = None,
    ) -> list[SearchDatabaseItem]:
        """Exclude database items from the same patient as the query."""
        database = self.search_database if database_items is None else database_items

        query_patient_id = query_item.metadata.get("patient_id")
        if query_patient_id is None:
            return list(database)

        return [
            item
            for item in database
            if item.metadata.get("patient_id") != query_patient_id
        ]

    def search(
        self,
        query_representation: RetrievalRepresentation,
        *,
        filter_same_patient: bool = True,
        **kwargs,
    ) -> SearchResult:
        """Run search for one query against the current database."""
        query_item = self.prepare_query(query_representation)

        database_items = self.search_database
        if filter_same_patient:
            database_items = self.filter_database_by_patient(
                query_item=query_item,
                database_items=database_items,
            )

        hits = self.rank(
            query_item=query_item,
            database_items=database_items,
            **kwargs,
        )

        return SearchResult(
            query_id=query_item.item_id,
            hits=hits,
            metadata=query_item.metadata,
        )

    def rank(
        self,
        query_item: SearchDatabaseItem,
        database_items: list[SearchDatabaseItem],
        **kwargs,
    ) -> list[SearchHit]:
        """Rank database items for one prepared query."""
        raise NotImplementedError

    def _validate_representations(
        self,
        representations: list[RetrievalRepresentation],
    ) -> None:
        """Validate that all representations are supported by this strategy."""
        if not self.supported_representation_kinds:
            raise ValueError(
                f"Search strategy '{self.name}' does not define "
                f"supported_representation_kinds."
            )

        for representation in representations:
            representation_kind = str(representation.representation_type).strip().lower()
            if representation_kind not in self.supported_representation_kinds:
                raise ValueError(
                    f"Search strategy '{self.name}' does not support representation kind "
                    f"'{representation_kind}'. Supported kinds: "
                    f"{sorted(self.supported_representation_kinds)}"
                )
