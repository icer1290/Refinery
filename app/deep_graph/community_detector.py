"""Community detection using Leiden algorithm.

Detects communities of related entities in the knowledge graph.
"""

import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.graph_store import graph_store
from app.deep_graph.state import Community
from app.models.orm_models import GraphEntity, GraphRelationship

logger = get_logger(__name__)
settings = get_settings()


class CommunityDetector:
    """Detects communities in the knowledge graph using Leiden algorithm.

    Uses igraph and leidenalg for community detection.
    Falls back to simple connected components if libraries not available.
    """

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if igraph and leidenalg are available."""
        try:
            import igraph
            import leidenalg
            self._has_leiden = True
        except ImportError:
            logger.warning(
                "igraph or leidenalg not available, "
                "community detection will use simple fallback"
            )
            self._has_leiden = False
        return self._has_leiden

    async def detect_communities(
        self,
        session: AsyncSession,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
    ) -> List[Community]:
        """Detect communities in the entity graph.

        Args:
            session: Database session
            entities: List of graph entities
            relationships: List of graph relationships

        Returns:
            List of detected communities
        """
        if not entities:
            return []

        logger.info(
            "Starting community detection",
            entity_count=len(entities),
            relationship_count=len(relationships),
        )

        if self._has_leiden:
            communities = await self._detect_with_leiden(
                session, entities, relationships
            )
        else:
            communities = await self._detect_simple(
                session, entities, relationships
            )

        logger.info(
            "Community detection complete",
            community_count=len(communities),
        )

        return communities

    async def _detect_with_leiden(
        self,
        session: AsyncSession,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
    ) -> List[Community]:
        """Detect communities using Leiden algorithm."""
        import igraph as ig
        import leidenalg as la

        # Build entity ID to index mapping
        entity_id_to_idx = {str(e.id): i for i, e in enumerate(entities)}
        idx_to_entity_id = {i: str(e.id) for i, e in enumerate(entities)}

        # Build edges list
        edges = []
        edge_weights = []
        for rel in relationships:
            src_id = str(rel.source_entity_id)
            tgt_id = str(rel.target_entity_id)

            if src_id in entity_id_to_idx and tgt_id in entity_id_to_idx:
                src_idx = entity_id_to_idx[src_id]
                tgt_idx = entity_id_to_idx[tgt_id]
                edges.append((src_idx, tgt_idx))
                edge_weights.append(rel.weight or 1.0)

        if not edges:
            # No edges, each entity is its own community
            return [
                Community(
                    id=str(uuid.uuid4()),
                    name=f"独立实体: {e.name}",
                    summary=f"独立实体，暂无已知关联",
                    entity_ids=[str(e.id)],
                    hub_entity_id=str(e.id),
                    article_ids=[str(aid) for aid in (e.article_ids or [])],
                    level=0,
                )
                for e in entities
            ]

        # Create igraph
        g = ig.Graph(n=len(entities), edges=edges, edge_attrs={"weight": edge_weights})

        # Run Leiden algorithm
        partition = la.find_partition(
            g,
            la.ModularityVertexPartition,
            weights="weight",
        )

        # Convert to Community objects
        communities = []
        for i, cluster in enumerate(partition):
            if len(cluster) == 0:
                continue

            cluster_entity_ids = [idx_to_entity_id[idx] for idx in cluster]
            cluster_entities = [e for e in entities if str(e.id) in cluster_entity_ids]

            # Find hub entity (highest degree in cluster)
            hub_idx = cluster[0]
            if len(cluster) > 1:
                degrees = [g.degree(idx) for idx in cluster]
                hub_idx = cluster[degrees.index(max(degrees))]

            hub_entity_id = idx_to_entity_id[hub_idx]
            hub_entity = next(
                (e for e in entities if str(e.id) == hub_entity_id),
                None
            )

            # Collect article IDs
            article_ids = set()
            for e in cluster_entities:
                for aid in (e.article_ids or []):
                    article_ids.add(str(aid))

            community = Community(
                id=str(uuid.uuid4()),
                name=f"社区 {i + 1}",
                summary=self._generate_community_summary(cluster_entities),
                entity_ids=cluster_entity_ids,
                hub_entity_id=hub_entity_id,
                article_ids=list(article_ids),
                level=0,
            )
            communities.append(community)

        return communities

    async def _detect_simple(
        self,
        session: AsyncSession,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
    ) -> List[Community]:
        """Simple community detection using connected components.

        Fallback when igraph/leidenalg are not available.
        """
        # Build adjacency list
        entity_ids = {str(e.id) for e in entities}
        adjacency: Dict[str, set] = defaultdict(set)

        for rel in relationships:
            src = str(rel.source_entity_id)
            tgt = str(rel.target_entity_id)
            if src in entity_ids and tgt in entity_ids:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)

        # Find connected components using BFS
        visited = set()
        communities = []

        for entity in entities:
            entity_id = str(entity.id)
            if entity_id in visited:
                continue

            # BFS to find connected component
            component = []
            queue = [entity_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if not component:
                continue

            # Create community
            component_entities = [
                e for e in entities if str(e.id) in component
            ]

            article_ids = set()
            for e in component_entities:
                for aid in (e.article_ids or []):
                    article_ids.add(str(aid))

            community = Community(
                id=str(uuid.uuid4()),
                name=f"社区 {len(communities) + 1}",
                summary=self._generate_community_summary(component_entities),
                entity_ids=component,
                hub_entity_id=component[0] if component else None,
                article_ids=list(article_ids),
                level=0,
            )
            communities.append(community)

        return communities

    def _generate_community_summary(
        self,
        entities: List[GraphEntity],
    ) -> str:
        """Generate a brief summary for a community.

        Args:
            entities: Entities in the community

        Returns:
            Summary string
        """
        if not entities:
            return "空社区"

        # Count by type
        type_counts: Dict[str, int] = defaultdict(int)
        for e in entities:
            type_counts[e.type] = (type_counts[e.type] or 0) + 1

        type_summary = ", ".join(
            f"{t}: {c}个" for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
        )

        # Get top entities by mention count
        top_entities = sorted(
            entities,
            key=lambda e: e.mention_count or 0,
            reverse=True
        )[:3]
        top_names = [e.name for e in top_entities]

        return f"包含{len(entities)}个实体（{type_summary}），核心实体：{', '.join(top_names)}"


# Singleton instance
community_detector = CommunityDetector()