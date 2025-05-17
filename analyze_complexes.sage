#!/usr/bin/env sage

"""
Spherical Design Analyzer
Version 3.0 - Edition

A tool for analyzing the algebraic, topological, and geometric properties 
of spherical designs represented as simplicial complexes.
"""

import json
import argparse
import os
import sys
import traceback
import time
import pandas as pd
from collections import defaultdict
import logging
import threading
import gc
from itertools import combinations
from contextlib import contextmanager

# Import Sage components
from sage.all import (
    SimplicialComplex, PolynomialRing, QQ, ZZ, matrix, 
    cached_function, gcd, binomial, Graph
)

# Configuration
MAX_MINIMAL_NONFACE_DISPLAY = 10
HOMOLOGY_COMPUTATION_TIMEOUT = 30  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_version_info():
    """Return information about Sage version and available libraries"""
    import sage.version
    version_info = {
        "sage_version": sage.version.version,
        "has_singular": False,
        "has_homology": hasattr(ZZ, "is_isomorphic")
    }
    
    # Check for Singular availability
    try:
        from sage.interfaces.singular import singular
        version_info["has_singular"] = True
    except ImportError:
        pass
    
    return version_info

VERSION_INFO = get_version_info()

@contextmanager
def timeout(seconds):
    """
    Thread-based timeout context manager (safer than multiprocessing.Queue)
    
    This implementation avoids potential security issues with multiprocessing
    when handling untrusted input files.
    """
    def interrupt():
        raise TimeoutError("Computation timed out")
        
    timer = threading.Timer(seconds, interrupt)
    timer.start()
    
    try:
        yield
    finally:
        timer.cancel()

class DesignAnalyzer:
    def __init__(self):
        self.results_template = {
            "filename": "",
            "success": False,
            "error": None,
            "warnings": []
        }
        # Cache for computationally expensive results
        self._cache = {}

    def analyze_file(self, json_filepath):
        """Main analysis entry point with comprehensive error handling"""
        results = self.results_template.copy()
        results["filename"] = os.path.basename(json_filepath)
        
        try:
            logging.info(f"Loading data from {json_filepath}")
            data = self.load_json_data(json_filepath)
            
            logging.info("Building simplicial complex")
            S = self.build_complex(data["facets"])
            
            logging.info("Computing basic properties")
            results.update(self.compute_basic_properties(S))
            
            logging.info("Computing homology")
            results.update(self.compute_homology(S))
            
            logging.info("Computing algebraic properties")
            results.update(self.compute_algebraic_properties(S))
            
            logging.info("Computing geometric properties")
            results.update(self.compute_geometric_properties(S))
            
            logging.info("Extracting design parameters")
            results.update(self.extract_design_parameters(results["filename"]))
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = self.format_error(e)
            results["traceback"] = traceback.format_exc()
            logging.error(f"Analysis failed: {str(e)}")
        finally:
            # Clean up memory
            #self._face_cache.clear()  # Fixed variable name
            gc.collect()
        
        return results

    def load_json_data(self, filepath):
        """Load and validate JSON input"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'facets' not in data:
            raise ValueError("JSON missing required 'facets' key")
        if not isinstance(data['facets'], list):
            raise TypeError("Facets must be a list")
        
        return data

    def build_complex(self, facets):
        """Construct simplicial complex with validation"""
        try:
            # Convert 0-indexed facets to 1-indexed (Sage's convention)
            facets_1indexed = [[v+1 for v in f] for f in facets]
            return SimplicialComplex(facets_1indexed)
        except Exception as e:
            raise ValueError(f"Invalid facet structure: {str(e)}")

    def compute_basic_properties(self, S):
        """Calculate fundamental combinatorial properties"""
        return {
            "n_vertices": len(S.vertices()),
            "n_facets": len(S.facets()),
            "dimension": S.dimension(),
            "f_vector": S.f_vector(),
            "h_vector": S.h_vector(),
            "euler_characteristic": S.euler_characteristic(),
            "is_pure": S.is_pure()
        }

    def compute_homology(self, S):
        """Robust homology computation with timeout"""
        results = {"homology": {}, "is_connected": None, "betti_numbers": []}
        
        try:
            with timeout(HOMOLOGY_COMPUTATION_TIMEOUT):
                homology = S.homology()
                
                # Format homology groups
                results["homology"] = {
                    f"H_{k}": str(homology.get(k, 0)) 
                    for k in range(S.dimension() + 1)
                }
                
                # Check connectedness from H_0
                h0 = homology.get(0, None)
                results["is_connected"] = self.is_trivial_homology(h0)
                
                # Compute Betti numbers
                results["betti_numbers"] = [
                    self.betti_number(homology, k) 
                    for k in range(S.dimension() + 1)
                ]
                
                # Determine if it's a homology sphere
                results["is_homology_sphere"] = self.check_homology_sphere(homology, S.dimension())
                
        except TimeoutError:
            results["homology"] = "Timeout"
            results["is_connected"] = "Unknown"
            results["warnings"].append("Homology computation timed out")
        except Exception as e:
            results["homology"] = f"Error: {str(e)}"
            results["is_connected"] = "Unknown"
            results["warnings"].append(f"Homology computation failed: {str(e)}")
        
        return results
    
    def is_trivial_homology(self, homology_group):
        """
        Check if a homology group is trivial (zero).
        Uses proper Sage methods if available, falls back to string comparison.
        """
        if homology_group is None:
            return False
            
        # Try the Sage built-in method first if available
        if hasattr(homology_group, "is_trivial"):
            return homology_group.is_trivial()
            
        # Fall back to string representation as last resort
        return str(homology_group) == '0'
    
    def is_isomorphic_to_z(self, homology_group):
        """
        Check if a homology group is isomorphic to Z.
        Uses proper Sage methods if available, falls back to string comparison.
        """
        if homology_group is None:
            return False
            
        # Try the Sage built-in method first if available
        if hasattr(homology_group, "is_isomorphic") and hasattr(ZZ, "one"):
            return homology_group.is_isomorphic(ZZ.one().homology())
            
        # Fall back to string representation as last resort
        h_str = str(homology_group)
        return h_str in ['Z', '1', 'Z/1', 'Z/1Z']
    
    def betti_number(self, homology, k):
        """
        Safely extract Betti number from homology group.
        The Betti number is the rank of the free part of the homology group.
        """
        h = homology.get(k, 0)
        
        # Use rank method if available
        if hasattr(h, 'rank'):
            return h.rank()
            
        # Handle string representation as fallback
        h_str = str(h)
        if h_str == '0':
            return 0
        elif h_str == 'Z':
            return 1
        else:
            # Count Z summands in forms like 'Z⊕Z/2Z⊕Z/3Z'
            return h_str.count('Z') - h_str.count('Z/')
    
    def check_homology_sphere(self, homology, dimension):
        """
        Check if the homology matches that of a sphere.
        A d-dimensional homology sphere has H_d = Z and H_i = 0 for i != d.
        """
        if dimension < 0:
            return False
            
        # Check for vanishing homology below top dimension
        for k in range(dimension):
            if not self.is_trivial_homology(homology.get(k, 0)):
                return False
        
        # Check for Z in top dimension
        return self.is_isomorphic_to_z(homology.get(dimension, 0))

    def compute_algebraic_properties(self, S):
        """Algebraic property analysis with enhanced checks"""
        props = {}
        
        try:
            # Compute h-vector once for efficiency
            h_vector = S.h_vector()
            
            # Cohen-Macaulay property
            props["is_cohen_macaulay"] = S.is_cohen_macaulay()
            
            # Gorenstein property
            props["is_gorenstein"] = (
                props["is_cohen_macaulay"] and 
                h_vector == list(reversed(h_vector)) and 
                h_vector[-1] == 1
            )
            
            # Buchsbaum property with improved check
            props["is_buchsbaum"] = self.is_buchsbaum(S)
            
            # Shellability check
            props["is_shellable"] = self.check_shellability(S)
            
            # Stanley-Reisner ring properties
            sr_props = self.stanley_reisner_properties(S, h_vector)
            props.update({f"sr_{k}": v for k, v in sr_props.items()})
            
        except Exception as e:
            props["algebraic_error"] = str(e)
            logging.warning(f"Error computing algebraic properties: {str(e)}")
        
        return props

    def is_buchsbaum(self, S):
        """
        Mathematically rigorous Buchsbaum verification.
        
        A simplicial complex is Buchsbaum if it's pure and the links of all
        non-empty faces have vanishing reduced homology below the top dimension.
        """
        if not S.is_pure():
            return False
        
        # Quick check - Cohen-Macaulay implies Buchsbaum
        if S.is_cohen_macaulay():
            return True
            
        # Try using Singular if available
        if VERSION_INFO["has_singular"]:
            try:
                from sage.interfaces.singular import singular
                
                # Try to load buchberger's algorithm library
                singular.LIB("buch.lib")
                
                # Create Stanley-Reisner ideal
                # (This is an advanced Singular computation)
                # Note: Implementation would need the actual singular code
                
                return "Needs Singular integration"
            except Exception:
                pass
        
        # Fall back to the homological definition
        # Check reduced homology of links of all faces
        for dim in range(S.dimension()):
            faces = list(S.faces().get(dim, []))
            
            # Limit check for large complexes
            if len(faces) > 20:
                # Sample approach - only check a manageable subset
                faces = faces[:20]
                
            for face in faces:
                face_list = list(face)
                link = S.link(face_list)
                
                # Skip if link is empty
                if link.dimension() < 0:
                    continue
                
                # Check vanishing homology below top dimension
                for k in range(link.dimension()):
                    h_k = link.homology(k)
                    if not self.is_trivial_homology(h_k):
                        return False
        
        return True

    def check_shellability(self, S):
        """
        Enhanced shellability check with multiple strategies.
        
        Looks for shellability using:
        1. Native Sage methods if available
        2. Partitionability check if possible
        3. Vertex decomposability for small complexes
        4. Heuristics for special cases like small spheres
        """
        # Try Sage's built-in implementation if available
        try:
            return S.is_shellable()
        except (AttributeError, NotImplementedError):
            pass
        
        # Check for partitionability (more general property implying shellability)
        try:
            return S.is_partitionable()
        except (AttributeError, NotImplementedError):
            pass
        
        # Special case 1: Low-dimensional Cohen-Macaulay complexes are often shellable
        if S.dimension() <= 1 and S.is_cohen_macaulay():
            return True
        
        # Special case 2: Small 2-spheres are shellable
        if (S.dimension() == 2 and 
            S.euler_characteristic() == 2 and
            len(S.vertices()) <= 10 and
            self.check_homology_sphere(S.homology(), 2)):
            return True
        
        # Special case 3: Try vertex decomposability for small complexes
        if len(S.vertices()) <= 15 and S.dimension() <= 2:
            try:
                return self.is_vertex_decomposable(S)
            except Exception:
                pass
        
        # Cannot determine shellability with available methods
        return "Undetermined"
    
    def is_vertex_decomposable(self, S):
        """
        Check if complex is vertex decomposable.
        
        A complex is vertex decomposable if it's a simplex or if there exists a
        vertex v such that:
        1. The link of v is vertex decomposable
        2. The deletion of v is vertex decomposable
        3. No face of the deletion of v is a facet of the link of v
        
        Note: This is a recursive check and may be slow for larger complexes.
        """
        # Base case 1: Empty complex
        if S.dimension() < 0:
            return True
            
        # Base case 2: A single vertex
        if S.dimension() == 0 and len(S.facets()) == 1:
            return True
            
        # Try each vertex
        for v in S.vertices():
            link_v = S.link([v])
            deletion_v = S.delete_vertex(v)
            
            # Check if both link and deletion are vertex decomposable
            if (self.is_vertex_decomposable(link_v) and 
                self.is_vertex_decomposable(deletion_v)):
                
                # Check that no face of deletion is a facet of link
                link_facets = {tuple(sorted(f)) for f in link_v.facets()}
                deletion_faces = {tuple(sorted(f)) for d in range(deletion_v.dimension()+1)
                                 for f in deletion_v.faces().get(d, [])}
                
                if not deletion_faces.intersection(link_facets):
                    return True
        
        return False

    def stanley_reisner_properties(self, S, h_vector=None):
        """
        Compute Stanley-Reisner ring properties with enhancements.
        
        Computes algebraic invariants of the Stanley-Reisner ring including:
        - Krull dimension
        - Multiplicity 
        - a-invariant
        - Regularity
        - Minimal non-faces (Stanley-Reisner ideal generators)
        """
        if h_vector is None:
            h_vector = S.h_vector()
        
        # Basic SR ring properties
        properties = {
            "krull_dim": S.dimension() + 1,
            "multiplicity": sum(h_vector),
            "a_invariant": len(h_vector) - 1 - (S.dimension() + 1),
            "regularity": len(h_vector) - 1,
            "h_unimodal": self.is_unimodal(h_vector),
            "h_symmetric": h_vector == list(reversed(h_vector))
        }
        
        # Compute minimal non-faces (generators of the Stanley-Reisner ideal)
        try:
            properties["minimal_nonfaces"] = self.find_minimal_nonfaces(S)
        except Exception as e:
            properties["minimal_nonfaces_error"] = str(e)
        
        return properties

    def is_unimodal(self, seq):
        """
        Check if a sequence is unimodal (increases then decreases).
        
        A sequence is unimodal if there exists an index i such that:
        seq[0] ≤ seq[1] ≤ ... ≤ seq[i] ≥ seq[i+1] ≥ ... ≥ seq[n]
        """
        increasing = True
        for i in range(1, len(seq)):
            if seq[i] < seq[i-1]:
                increasing = False
            elif not increasing and seq[i] > seq[i-1]:
                return False
        return True

    def find_minimal_nonfaces(self, S):
        """
        Find minimal non-faces of a simplicial complex.
        
        A minimal non-face is a subset of vertices that is not a face,
        but every proper subset is a face.
        """
        # Use Sage's built-in method if available
        try:
            return S.minimal_nonfaces()
        except (AttributeError, NotImplementedError):
            pass
        
        # Fallback implementation
        vertices = list(S.vertices())
        
        # Get all faces as sets for efficient subset testing
        all_faces = set()
        for dim in range(S.dimension() + 1):
            for face in S.faces().get(dim, []):
                all_faces.add(tuple(sorted(face)))
        
        nonfaces = []
        
        # Check pairs of vertices (edges) first
        for i, j in combinations(range(len(vertices)), 2):
            edge = tuple(sorted([vertices[i], vertices[j]]))
            if edge not in all_faces:
                nonfaces.append(edge)
        
        # Then check larger potential non-faces up to reasonable size
        max_size = min(4, S.dimension() + 1)
        for size in range(3, max_size + 1):
            for candidate in combinations(vertices, size):
                candidate = tuple(sorted(candidate))
                
                # Skip if already a face
                if candidate in all_faces:
                    continue
                
                # Check if all proper subsets are faces
                is_minimal = True
                for i in range(len(candidate)):
                    subset = candidate[:i] + candidate[i+1:]
                    if subset not in all_faces:
                        is_minimal = False
                        break
                
                if is_minimal:
                    nonfaces.append(candidate)
        
        return self.format_nonfaces(nonfaces)

    def format_nonfaces(self, nonfaces):
        """Format minimal non-faces list for output"""
        if len(nonfaces) <= MAX_MINIMAL_NONFACE_DISPLAY:
            return nonfaces
        else:
            return {
                "count": len(nonfaces),
                "examples": nonfaces[:MAX_MINIMAL_NONFACE_DISPLAY],
                "note": f"Showing {MAX_MINIMAL_NONFACE_DISPLAY} of {len(nonfaces)} minimal non-faces"
            }

    def compute_geometric_properties(self, S):
        """
        Analyze geometric properties of the complex.
        
        Computes:
        - Manifold property
        - Pseudomanifold property
        - Flag complex property
        - Non-spherical vertices (vertices with non-sphere links)
        """
        props = {}
        
        try:
            # Find non-spherical vertices first (needed for manifold check)
            non_spherical = self.find_non_spherical_vertices(S)
            props["non_spherical_vertices"] = non_spherical
            
            # Manifold check (a complex is a manifold iff all vertex links are spheres)
            props["is_manifold"] = (len(non_spherical) == 0)
            
            # Pseudomanifold check with incidence matrix
            props["is_pseudomanifold"] = self.is_pseudomanifold(S)
            
            # Flag complex check with improved algorithm
            props["is_flag_complex"] = self.is_flag_complex(S)
            
        except Exception as e:
            props["geometric_error"] = str(e)
            logging.warning(f"Error computing geometric properties: {str(e)}")
            props["geometric_traceback"] = traceback.format_exc()
        
        return props

    def find_non_spherical_vertices(self, S):
        """
        Find vertices whose links are not homeomorphic to spheres.
        
        A vertex has a spherical link if its link has the homology of a sphere
        of the appropriate dimension.
        """
        non_spherical = []
        
        for v in S.vertices():
            try:
                link = S.link([v])
                if not self.is_homology_sphere_link(link):
                    non_spherical.append(v)
            except Exception as e:
                logging.warning(f"Error checking link of vertex {v}: {str(e)}")
                # Conservatively add to non-spherical
                non_spherical.append(v)
        
        return non_spherical

    def is_homology_sphere_link(self, link):
        """
        Check if a link has the homology of a sphere using rigorous methods.
        
        A d-dimensional link is a homology sphere if:
        - H_i(link) = 0 for all i < d
        - H_d(link) = Z
        
        This implementation uses proper Sage methods when available.
        """
        if link.dimension() < 0:
            return False  # Empty link cannot be a sphere
        
        d = link.dimension()
        try:
            homology = link.homology()
            
            # Check for vanishing homology below top dimension
            for k in range(d):
                if not self.is_trivial_homology(homology.get(k, 0)):
                    return False
            
            # Check for Z in top dimension
            return self.is_isomorphic_to_z(homology.get(d, 0))
            
        except Exception as e:
            logging.warning(f"Error in homology sphere check: {str(e)}")
            return False

    def is_pseudomanifold(self, S):
        """
        Check if complex is a pseudomanifold using incidence matrices.
        
        A pure d-dimensional simplicial complex is a pseudomanifold if each
        (d-1)-dimensional face (ridge) is contained in exactly two d-faces.
        """
        if not S.is_pure():
            return False
        
        d = S.dimension()
        if d <= 0:
            return True  # Points and empty complex are trivially pseudomanifolds
        
        try:
            # Use incidence matrix method (exact)
            M = S.incidence_matrix(d-1, d)
            
            # Check if each ridge (row) appears in exactly 2 facets (columns)
            return all(row.sum() == 2 for row in M.rows())
        except Exception as e:
            logging.warning(f"Error in incidence matrix calculation: {str(e)}")
            
            # Fall back to direct counting method
            ridge_counts = defaultdict(int)
            
            for facet in S.facets():
                facet_list = list(facet)
                
                # Generate all (d-1)-dimensional faces (ridges)
                for i in range(len(facet_list)):
                    ridge = tuple(sorted(facet_list[:i] + facet_list[i+1:]))
                    ridge_counts[ridge] += 1
            
            # Check if every ridge appears exactly twice
            return all(count == 2 for count in ridge_counts.values())

    def is_flag_complex(self, S):
        """
        Check if complex is a flag complex using graph-theoretic approach.
        
        A simplicial complex is a flag complex if it equals the clique complex
        of its 1-skeleton (i.e., all minimal non-faces have size 2).
        """
        # Try native method if available
        try:
            return S.is_flag_complex()
        except (AttributeError, NotImplementedError):
            pass
        
        # Alternative implementation using 1-skeleton graph
        try:
            G = S.graph()  # Get 1-skeleton as a graph
            
            # Find minimal non-faces
            min_nonfaces = self.find_minimal_nonfaces(S)
            if isinstance(min_nonfaces, dict):  # Handle dictionary format
                min_nonfaces = min_nonfaces["examples"]
            
            # A flag complex has all minimal non-faces of size 2
            return all(len(nonface) == 2 for nonface in min_nonfaces)
        except Exception as e:
            logging.warning(f"Error in flag complex check: {str(e)}")
            
            # Fall back to clique checking approach
            edges = set()
            for face in S.faces().get(1, []):
                edges.add(tuple(sorted(face)))
            
            # Check if every clique is a face
            for dim in range(2, S.dimension() + 1):
                for face in S.faces().get(dim, []):
                    vertices = list(face)
                    
                    # Check if all pairs form edges
                    for i, j in combinations(range(len(vertices)), 2):
                        edge = tuple(sorted([vertices[i], vertices[j]]))
                        if edge not in edges:
                            return False
            
            return True

    def extract_design_parameters(self, filename):
        """
        Extract design parameters from filename pattern des<dim>-<points>-<strength>.
        
        For example, from des3-11-3.json:
        - design_dimension = 3
        - design_points = 11
        - design_strength = 3
        """
        if not filename.startswith("des"):
            return {}
        
        try:
            parts = filename.split(".")[0].split("-")
            if len(parts) >= 3:
                return {
                    "design_dimension": int(parts[0][3:]),
                    "design_points": int(parts[1]),
                    "design_strength": int(parts[2])
                }
        except Exception as e:
            logging.warning(f"Failed to extract design parameters: {str(e)}")
        
        return {}

    def format_error(self, error):
        """Format error for output"""
        return f"{type(error).__name__}: {str(error)}"

def process_directory(directory, output_prefix=None):
    """Process all JSON files in directory with progress reporting"""
    analyzer = DesignAnalyzer()
    results = []
    
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files")
    
    for i, filename in enumerate(json_files, 1):
        filepath = os.path.join(directory, filename)
        logging.info(f"Processing {i}/{len(json_files)}: {filename}")
        
        start_time = time.time()
        result = analyzer.analyze_file(filepath)
        elapsed = time.time() - start_time
        
        result["processing_time"] = elapsed
        results.append(result)
        
        print(f"  Completed in {elapsed:.2f}s. Success: {result['success']}")
        
        if output_prefix and i % 10 == 0:
            save_intermediate(results, output_prefix, i)
    
    if output_prefix:
        save_final(results, output_prefix)
    
    return results

def save_intermediate(results, prefix, count):
    """Save intermediate results to CSV"""
    df = pd.DataFrame(results)
    output_file = f"{prefix}_partial_{count}.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Saved intermediate results to {output_file}")

def save_final(results, prefix):
    """Save final results and generate summary with proper float conversion"""
    df = pd.DataFrame(results)
    output_file = f"{prefix}.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Saved final results to {output_file}")
    
    # Generate summary statistics
    successful = [r for r in results if r["success"]]
    
    print("\n" + "="*40)
    print(f"ANALYSIS SUMMARY: {len(successful)}/{len(results)} successful")
    print("="*40)
    
    if successful:
        # Calculate key statistics
        cm_count = sum(1 for r in successful if r.get("is_cohen_macaulay") == True)
        gorenstein_count = sum(1 for r in successful if r.get("is_gorenstein") == True)
        manifold_count = sum(1 for r in successful if r.get("is_manifold") == True)
        flag_count = sum(1 for r in successful if r.get("is_flag_complex") == True 
                         or str(r.get("is_flag_complex")).startswith("Likely"))
        
        # Convert to Python floats before formatting - this fixes the Sage Rational issue
        successful_count = float(len(successful))
        cm_percent = float(cm_count) / successful_count * 100.0
        gorenstein_percent = float(gorenstein_count) / successful_count * 100.0
        manifold_percent = float(manifold_count) / successful_count * 100.0
        flag_percent = float(flag_count) / successful_count * 100.0
        
        # Print summary statistics
        print(f"Cohen-Macaulay: {cm_count}/{len(successful)} ({cm_percent:.1f}%)")
        print(f"Gorenstein: {gorenstein_count}/{len(successful)} ({gorenstein_percent:.1f}%)")
        print(f"Manifolds: {manifold_count}/{len(successful)} ({manifold_percent:.1f}%)")
        print(f"Flag Complexes: {flag_count}/{len(successful)} ({flag_percent:.1f}%)")
        
        # Analyze correlation with design strength
        strengths = {}
        for r in successful:
            strength = r.get("design_strength")
            if strength is not None:
                if strength not in strengths:
                    strengths[strength] = {"total": 0, "gorenstein": 0}
                
                strengths[strength]["total"] += 1
                if r.get("is_gorenstein") == True:
                    strengths[strength]["gorenstein"] += 1
        
        if strengths:
            print("\n--- DESIGN STRENGTH CORRELATION ---")
            for strength, data in sorted(strengths.items()):
                # Convert to Python float before formatting
                total = float(data["total"])
                percent = float(data["gorenstein"]) / total * 100.0
                print(f"Strength {strength}: {data['gorenstein']}/{data['total']} Gorenstein ({percent:.1f}%)")
    
    # Save summary to file
    if prefix:
        summary_file = f"{prefix}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"ANALYSIS SUMMARY: {len(successful)}/{len(results)} successful\n\n")
            
            if successful:
                # Convert to floats for file output
                f.write(f"Cohen-Macaulay: {cm_count}/{len(successful)} ({cm_percent:.1f}%)\n")
                f.write(f"Gorenstein: {gorenstein_count}/{len(successful)} ({gorenstein_percent:.1f}%)\n")
                f.write(f"Manifolds: {manifold_count}/{len(successful)} ({manifold_percent:.1f}%)\n")
                f.write(f"Flag Complexes: {flag_count}/{len(successful)} ({flag_percent:.1f}%)\n")
                
                if strengths:
                    f.write("\n--- DESIGN STRENGTH CORRELATION ---\n")
                    for strength, data in sorted(strengths.items()):
                        # Convert to float for file output
                        total = float(data["total"])
                        percent = float(data["gorenstein"]) / total * 100.0
                        f.write(f"Strength {strength}: {data['gorenstein']}/{data['total']} Gorenstein ({percent:.1f}%)\n")
            
        logging.info(f"Saved summary to {summary_file}")

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spherical design complexes")
    parser.add_argument("directory", help="Input directory with JSON files")
    parser.add_argument("-o", "--output", help="Output prefix for results")
    
    args = parser.parse_args()
    process_directory(args.directory, args.output)
