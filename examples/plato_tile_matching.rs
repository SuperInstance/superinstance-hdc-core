//! # PLATO / MUD Tile Matching Example
//!
//! Demonstrates how HDC can match agent-generated tiles against a
//! knowledge base of "good" tiles — for quality filtering,
//! duplicate detection, and semantic routing.
//!
//! ## Use Case
//!
//! ZC agents produce ~8300 tiles/day. Oracle1 needs to:
//! 1. Detect duplicates (same concept, different wording)
//! 2. Route tiles to the right PLATO room (semantic classification)
//! 3. Flag low-quality tiles (outliers)
//!
//! HDC solves all three in <1ms per tile.

use superinstance_hdc_core as hdc;

/// A PLATO tile, encoded as a 1024-bit hypervector.
#[derive(Debug, Clone)]
pub struct TileVector {
    pub topic: String,
    pub agent: String,
    pub hypervector: hdc::HyperVector,
}

/// Build a tile vector from the tile's content.
pub fn tile_to_vector(topic: &str, agent: &str, content: &str, seed: u64) -> TileVector {
    // Bundle topic + agent + content into one concept mask
    let hv = hdc::hdc::bundle_words(&[topic, agent, content], seed);
    TileVector {
        topic: topic.to_string(),
        agent: agent.to_string(),
        hypervector: hv,
    }
}

/// Find duplicate tiles (similarity > 0.95).
pub fn find_duplicates(tiles: &[TileVector], threshold: f64) -> Vec<(usize, usize, f64)> {
    let mut duplicates = Vec::new();
    for i in 0..tiles.len() {
        for j in (i + 1)..tiles.len() {
            let sim = tiles[i].hypervector.similarity(&tiles[j].hypervector);
            if sim >= threshold {
                duplicates.push((i, j, sim));
            }
        }
    }
    duplicates
}

/// Route a tile to the best-matching room.
pub fn route_to_room(
    tile: &TileVector,
    rooms: &[(String, hdc::HyperVector)],
    threshold: f64,
) -> Option<String> {
    let mut best_room = None;
    let mut best_sim = 0.0;

    for (name, room_hv) in rooms {
        let sim = tile.hypervector.similarity(room_hv);
        if sim > best_sim {
            best_sim = sim;
            best_room = Some(name.clone());
        }
    }

    if best_sim >= threshold {
        best_room
    } else {
        None // No good match — flag for review
    }
}

/// Detect outlier tiles (low similarity to all known good tiles).
pub fn detect_outliers(
    tiles: &[TileVector],
    good_examples: &[hdc::HyperVector],
    threshold: f64,
) -> Vec<usize> {
    let mut outliers = Vec::new();
    for (i, tile) in tiles.iter().enumerate() {
        let max_sim = good_examples
            .iter()
            .map(|good| tile.hypervector.similarity(good))
            .fold(0.0, f64::max);

        if max_sim < threshold {
            outliers.push(i);
        }
    }
    outliers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_duplicates() {
        let seed = 0xDEAD;
        let tiles = vec![
            tile_to_vector("flux", "zc-scholar", "constraint checking on GPU", seed),
            tile_to_vector("flux", "zc-scholar", "constraint checking on GPU", seed), // duplicate
            tile_to_vector("hdc", "zc-weaver", "hyperdimensional vectors", seed),
        ];

        let dups = find_duplicates(&tiles, 0.95);
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].0, 0);
        assert_eq!(dups[0].1, 1);
    }

    #[test]
    fn test_room_routing() {
        let seed = 0xDEAD;
        let rooms = vec![
            ("harbor".to_string(), hdc::HyperVector::from_text("blockers bugs p0", seed)),
            ("forge".to_string(), hdc::HyperVector::from_text("build css design", seed)),
            ("tide-pool".to_string(), hdc::HyperVector::from_text("research trends zc", seed)),
        ];

        let tile = tile_to_vector("research", "zc-scholar", "new flux instruction found", seed);
        let room = route_to_room(&tile, &rooms, 0.5);
        assert_eq!(room, Some("tide-pool".to_string()));
    }

    #[test]
    fn test_outlier_detection() {
        let seed = 0xDEAD;
        let good = vec![
            hdc::HyperVector::from_text("constraint safety formal verification", seed),
        ];

        let tiles = vec![
            tile_to_vector("safety", "zc-scholar", "proving memory safety", seed),
            tile_to_vector("cats", "zc-bard", "look at this cat photo", seed), // outlier
        ];

        let outliers = detect_outliers(&tiles, &good, 0.4);
        assert_eq!(outliers, vec![1]);
    }
}
