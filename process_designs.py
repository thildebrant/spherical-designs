import os
import glob
import numpy as np
import json
import re
import warnings
from scipy.spatial import ConvexHull, distance
from scipy.special import gamma
from fractions import Fraction # For exact representation
import math # For acos, degrees

# Helper function equivalent to MATLAB's uniquetol with 'ByRows'
# This is a simplified version and might not be a perfect match for all edge cases
# of MATLAB's uniquetol, especially with complex tolerances or dimensions beyond 3.
def uniquetol_by_rows(A, tol):
    """
    Finds unique rows in a 2D numpy array within a given tolerance.

    Args:
        A (np.ndarray): Input array (NxM).
        tol (float): Tolerance for considering elements equal.

    Returns:
        tuple: (unique_rows, original_indices, inverse_indices)
               unique_rows (np.ndarray): The unique rows.
               original_indices (np.ndarray): Indices of unique rows in the original array.
               inverse_indices (np.ndarray): Indices such that unique_rows[inverse_indices] == A.
    """
    # Sort rows to bring potentially close rows together
    # This helps, but doesn't guarantee that all close points are adjacent
    # A more robust method might involve clustering or KD-trees, but this
    # is a reasonable approximation for many cases.
    sorted_indices = np.lexsort(A.T)
    sorted_A = A[sorted_indices]

    # Find where the difference between adjacent rows is greater than tolerance
    # Check element-wise difference and then take the maximum absolute difference across columns
    diffs = np.diff(sorted_A, axis=0)
    # Check if the maximum absolute difference in any row pair is greater than tol
    is_different = np.max(np.abs(diffs), axis=1) > tol

    # The first row is always unique in the sorted list
    unique_mask = np.concatenate(([True], is_different))

    unique_sorted_indices = sorted_indices[unique_mask]
    unique_rows = A[unique_sorted_indices]

    # Build inverse_indices
    # This part is tricky. We need to map each original row back to its unique representative.
    # A simple way is to iterate through original rows and find which unique row it's close to.
    # This can be slow for large datasets.
    inverse_indices = np.zeros(A.shape[0], dtype=int)
    for i in range(A.shape[0]):
        # Find the index of the unique row closest to the current original row
        # We use cdist for pairwise distances
        distances_to_unique = distance.cdist([A[i]], unique_rows)
        closest_unique_idx = np.argmin(distances_to_unique)
        inverse_indices[i] = closest_unique_idx

    # The original_indices are the indices in the *original* array A
    # that correspond to the first occurrence of each unique row.
    # Since we used sorted_indices and unique_mask, unique_sorted_indices
    # gives us the indices in the original array A that are the first unique rows
    # after sorting. These are the original indices we need.
    original_indices = unique_sorted_indices

    return unique_rows, original_indices, inverse_indices

# Helper function equivalent to MATLAB's writeJSONsimp
def write_json_simp(name, faces, vertices=None):
    """
    Writes facets and optionally vertices to a simple JSON file.

    Args:
        name (str): The output filename.
        faces (np.ndarray): Fx3 array of face indices (1-based).
        vertices (np.ndarray, optional): Nx3 array of vertex coordinates. Defaults to None.
    """
    # Convert 1-based indices to 0-based for JSON
    S = {'facets': (faces - 1).tolist()}
    if vertices is not None and vertices.size > 0:
        S['vertices'] = vertices.tolist()

    try:
        with open(name, 'w') as f:
            # Use json.dumps with indent for pretty printing
            json.dump(S, f, indent=4)
    except IOError as e:
        warnings.warn(f"Could not write JSON file {name}: {e}")

# Helper function equivalent to MATLAB's edgesFromFaces
def edges_from_faces(F):
    """
    Extracts unique edges from a face list.

    Args:
        F (np.ndarray): Fx3 array of face indices.

    Returns:
        np.ndarray: Ex2 array of unique edge indices (sorted within each edge).
    """
    # Reshape faces to get pairs of vertices for each edge
    # Edges are (v1, v2), (v2, v3), (v3, v1) for each face (v1, v2, v3)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    # Sort vertex indices within each edge to handle (v1, v2) and (v2, v1) as the same edge
    sorted_edges = np.sort(edges, axis=1)
    # Find unique rows (edges)
    unique_edges = np.unique(sorted_edges, axis=0)
    return unique_edges

# Helper function equivalent to MATLAB's minAngleInMesh
def min_angle_in_mesh(V, F):
    """
    Calculates the minimum internal angle (in degrees) of triangles in a mesh.

    Args:
        V (np.ndarray): Nx3 array of vertex coordinates.
        F (np.ndarray): Fx3 array of face indices (1-based).

    Returns:
        float: The minimum angle in degrees, or NaN if no valid angles.
    """
    if F.size == 0:
        return np.nan

    min_angle_deg = 180.0
    valid_angles_found = False

    for face_indices in F:
        tri = V[face_indices - 1, :]  # Convert to 0-based indices
        D2 = distance.cdist(tri, tri, metric='sqeuclidean')  # Squared distances

        for j in range(3):
            u = (j + 1) % 3
            v = (j + 2) % 3

            # Compute cosine of angle using law of cosines (corrected formula)
            with np.errstate(divide='ignore', invalid='ignore'):
                numerator = D2[j, u] + D2[j, v] - D2[u, v]
                denominator = 2 * np.sqrt(D2[j, u] * D2[j, v])
                cosA = numerator / denominator
                cosA = np.clip(cosA, -1.0, 1.0)

            if np.isnan(cosA):
                continue  # Skip invalid/undefined angles

            valid_angles_found = True
            angle_deg = math.degrees(math.acos(cosA))
            min_angle_deg = min(min_angle_deg, angle_deg)

    return min_angle_deg if valid_angles_found else np.nan

# Helper function equivalent to MATLAB's qualityMetrics
def quality_metrics(V, F):
    """
    Calculates a few lightweight mesh/point-set quality indicators.

    Args:
        V (np.ndarray): Nx3 array of vertex coordinates.
        F (np.ndarray): Fx3 array of face indices (1-based).

    Returns:
        dict: A dictionary containing quality metrics.
              Keys: 'eulerDefect', 'minEdge', 'maxEdge', 'minAngle', 'AR_max'.
    """
    q = {
        'eulerDefect': np.nan,
        'minEdge': np.nan,
        'maxEdge': np.nan,
        'minAngle': np.nan,
        'AR_max': np.nan
    }

    if F.size == 0:
        return q

    # Euler characteristic: V - E + F - 2 = 0 for a simple sphere
    # For a triangulation of a sphere, V - E + F should be 2.
    # Euler defect is (V - E + F) - 2.
    E = edges_from_faces(F)
    q['eulerDefect'] = V.shape[0] - E.shape[0] + F.shape[0] - 2

    # Edge lengths
    # Get coordinates of edge vertices (adjusting for 0-based indexing)
    edge_vertices = V[E - 1, :]
    # Calculate lengths using numpy.linalg.norm
    L = np.linalg.norm(edge_vertices[:, 0, :] - edge_vertices[:, 1, :], axis=1)
    q['minEdge'] = np.min(L) if L.size > 0 else np.nan
    q['maxEdge'] = np.max(L) if L.size > 0 else np.nan

    # Triangle aspect ratios (longest/shortest edge per face)
    AR = []
    for face_indices in F:
        tri = V[face_indices - 1, :] # Adjust for 0-based indexing
        # Calculate pairwise distances within the triangle
        d = distance.pdist(tri) # Returns a condensed distance matrix (3 distances for a triangle)
        if np.min(d) > 0: # Avoid division by zero for degenerate edges
            AR.append(np.max(d) / np.min(d))
        else:
             # Handle degenerate triangles, maybe assign a very high AR or NaN
             # For now, we'll skip adding this AR, which means max(AR) won't consider it.
             # Depending on requirements, you might want to handle this differently.
             pass # Or AR.append(np.inf)

    q['AR_max'] = np.max(AR) if AR else np.nan # Handle case with no valid ARs

    # Minimum internal angle
    q['minAngle'] = min_angle_in_mesh(V, F)

    return q

# Helper function equivalent to MATLAB's strengthResidual
def strength_residual(P, t, tol=1e-15):
    """
    Verifies the spherical t-design condition (moments up to degree t).

    Args:
        P (np.ndarray): Nx3 array of points on the sphere.
        t (int): The design strength to check.
        tol (float, optional): Tolerance for the moment error. Defaults to 1e-15.

    Returns:
        float: The worst absolute error, or NaN if t is negative.
    """
    if t < 0:
        return np.nan

    # Robust normalization - ensure points are on the unit sphere
    norms = np.linalg.norm(P, axis=1)
    # Avoid division by zero for points at the origin
    P_normalized = P / norms[:, np.newaxis]
    # Handle points at the origin if necessary (they don't contribute to the integral)
    # For a spherical design, points should ideally be on the sphere, so this might not be needed
    # if input data is guaranteed to be normalized or non-zero.
    # P_normalized[norms == 0] = 0 # Or handle as an error/warning if points are expected on sphere

    max_err = 0.0

    # Gamma function values up to t
    g = [gamma((n + 1) / 2) for n in range(t + 1)]
    g0 = 1 / (4 * np.pi) # Normalization constant for the spherical integral

    # Iterate through all combinations of exponents (i, j, k) such that i+j+k <= t
    for i in range(t + 1):
        for j in range(t - i + 1):
            for k in range(t - i - j + 1):
                # Calculate the moment for the given exponents (i, j, k) from the points
                # mean(x^i * y^j * z^k)
                m_pts = np.mean(P_normalized[:, 0]**i * P_normalized[:, 1]**j * P_normalized[:, 2]**k)

                # Calculate the theoretical moment for the unit sphere
                # The integral of x^i y^j z^k over the unit sphere is zero if any exponent is odd.
                # If all exponents are even, there's a specific formula involving Gamma functions.
                if i % 2 != 0 or j % 2 != 0 or k % 2 != 0:
                    m_sphere = 0.0
                else:
                    # Formula for even exponents:
                    # Integral(x^2a y^2b z^2c) / Area(Sphere) = (Gamma(a+1/2)Gamma(b+1/2)Gamma(c+1/2)) / (2*pi*Gamma(a+b+c+3/2))
                    # Our exponents are i, j, k. So a=i/2, b=j/2, c=k/2.
                    # The formula in the MATLAB code seems slightly different, let's follow that one
                    # mSphere = g0 * 2 * g(i+1) * g(j+1) * g(k+1) / gamma((i+j+k+3)/2)
                    # Let's re-check the formula based on standard spherical harmonics integrals.
                    # The integral of x^i y^j z^k over the unit sphere is:
                    # 0 if any i, j, k is odd
                    # 2 * Gamma((i+1)/2) * Gamma((j+1)/2) * Gamma((k+1)/2) / Gamma((i+j+k)/2 + 3/2)
                    # when i, j, k are all even.
                    # The average moment is this integral divided by the surface area (4*pi).
                    # So, mSphere = (2 * Gamma((i+1)/2) * Gamma((j+1)/2) * Gamma((k+1)/2) / Gamma((i+j+k)/2 + 3/2)) / (4*pi)
                    # mSphere = Gamma((i+1)/2) * Gamma((j+1)/2) * Gamma((k+1)/2) / (2*pi * Gamma((i+j+k)/2 + 3/2))
                    # This matches the structure of the MATLAB code's formula if g(n+1) = Gamma((n+1)/2).
                    # Let's use the MATLAB formula directly as it's provided.
                    # g is pre-calculated where g[n] = Gamma((n+1)/2).
                    # So g[i] = Gamma((i+1)/2), g[j] = Gamma((j+1)/2), g[k] = Gamma((k+1)/2).
                    # The formula uses g(i+1), g(j+1), g(k+1) which corresponds to Gamma((i+1+1)/2), etc.
                    # This seems off based on standard integrals. Let's assume the MATLAB g() indexing is 1-based or
                    # there's a specific convention being used. Given the g definition `g = arrayfun(@(n) gamma((n+1)/2), 0:t);`,
                    # g[0] = gamma(1/2), g[1] = gamma(1), g[2] = gamma(3/2), etc.
                    # So g(i+1) in MATLAB (1-based index) corresponds to g[i] in 0-based Python index.
                    # The formula `g0*2* g(i+1)*g(j+1)*g(k+1) / gamma((i+j+k+3)/2)` becomes
                    # `g0 * 2 * g[i] * g[j] * g[k] / gamma((i+j+k)/2 + 3/2)` in Python.
                    # Let's use the pre-calculated `g` array correctly.
                    # The MATLAB code uses `g(i+1)` which means the (i+1)-th element of `g`.
                    # Since `g` is indexed from 0 to t, `g(i+1)` corresponds to `g[i]`.
                    # So the formula is `g0 * 2 * g[i] * g[j] * g[k] / gamma((i+j+k)/2 + 3/2)`.
                    # Let's recalculate g based on the indices i, j, k directly for clarity.
                    # This avoids potential off-by-one errors with the pre-calculated array.
                    m_sphere = g0 * 2 * gamma((i + 1) / 2) * gamma((j + 1) / 2) * gamma((k + 1) / 2) / gamma((i + j + k) / 2 + 3 / 2)

                err = abs(m_pts - m_sphere)
                max_err = max(max_err, err)

    return max_err

def process_designs(folder=None, tol=1e-12, do_exact=False):
    """
    Batch loader & checker for Sloane spherical designs.

    Args:
        folder (str, optional): Directory containing des*.txt files. Defaults to current directory.
        tol (float, optional): Duplicate / rational conversion tolerance. Defaults to 1e-12.
        do_exact (bool, optional): If True, convert coordinates to sympy symbolic. Defaults to False.

    Returns:
        list: A list of dictionaries, each containing data for a design file.
              Each dictionary contains keys: 'file', 'ptsFloat', 'ptsSym',
              'faces', 'volume', 'q'.
    """
    if folder is None:
        folder = os.getcwd()

    # Find design files
    search_pattern = os.path.join(folder, 'des*.txt')
    files = glob.glob(search_pattern)
    n_files = len(files)

    if n_files == 0:
        raise FileNotFoundError(f'No des*.txt files found in "{folder}".')

    print(f'▶ Processing {n_files} design files (exact = {do_exact})')

    data = []

    for k, fname in enumerate(files):
        file_info = os.path.basename(fname) # Get just the filename

        # -------------------------------------------------------------- load
        try:
            # Use genfromtxt to handle potential missing values or inconsistent rows more gracefully
            # Ensure we only read the first 3 columns and specify the comma delimiter
            C = np.genfromtxt(fname, usecols=(0, 1, 2), delimiter=',')
            # Check if the loaded data is valid (not empty and has 3 columns)
            if C.ndim != 2 or C.shape[1] != 3:
                 warnings.warn(f'File {file_info} is not a 3-column numerical matrix—skipped.')
                 continue
        except Exception as e:
            warnings.warn(f'Could not read file {file_info}: {e}—skipped.')
            continue

        # ----------------------------------------------------- deduplicate
        # Use the custom uniquetol_by_rows function
        V, _, _ = uniquetol_by_rows(C, tol)

        # ----------------------------------------------------- exact option
        pts_sym = []
        if do_exact:
            # Convert floating point coordinates to rational numbers using fractions.Fraction
            # This is an alternative to sympy.sympify(rat(C(j), tol))
            # You might need sympy if you need symbolic manipulation beyond just representation.
            # If sympy is required:
            # from sympy import sympify, Rational
            # pts_sym = np.empty(C.shape, dtype=object) # Use object dtype for sympy objects
            # for r in range(C.shape[0]):
            #     for c in range(C.shape[1]):
            #         # Use sympify(Fraction(value).limit_denominator(limit)) if needed
            #         # Or sympify(rat(C[r, c], tol)) if a direct rat equivalent is implemented
            #         # For now, let's use fractions.Fraction
            pts_sym = [[Fraction(val).limit_denominator() for val in row] for row in C]


        # ----------------------------------------------------- convex hull
        faces = []
        volume = np.nan # Initialize volume as NaN

        # Check if there are enough non-collinear points for a 3D convex hull
        # A 3D convex hull requires at least 4 points that are not coplanar.
        # Checking rank == 3 is a good heuristic for non-coplanarity of the unique points V.
        
        if V.shape[0] >= 4 and np.linalg.matrix_rank(V) == 3:
            try:
                hull = ConvexHull(V)
                faces = hull.simplices + 1  # NumPy array
                volume = hull.volume
            except Exception as e:
                warnings.warn(f'Convex hull failed for {file_info}: {e}')
                faces = np.array([], dtype=int)  # Empty array instead of list
                volume = np.nan
        else:
            faces = np.array([], dtype=int)  # Empty array instead of list
            volume = 0.0

        # Write convex hull to JSON file
        json_file = file_info.replace('.txt', '.json')
        write_json_simp(os.path.join(folder, json_file), faces, V) # Pass V to include vertices

        # ------------------------------------------------ quality metrics
        q = quality_metrics(V, faces)

        # —— parse t-strength from the filename (des3-14-4.txt → t = 4)
        # Regex to find the last number in the filename after a '-' or '_'
        match = re.search(r'des\d+[-_]\d+[-_](\d+)', file_info)
        if match:
            try:
                t_file = int(match.group(1))
                q['maxMomentErr'] = strength_residual(V, t_file, tol)
            except ValueError:
                warnings.warn(f'Could not parse t-strength from filename {file_info}.')
                q['maxMomentErr'] = np.nan
        else:
            q['maxMomentErr'] = np.nan

        # ----------------------------------------------------- data record
        data_record = {
            'file': file_info,
            'ptsFloat': C,
            'ptsSym': pts_sym if do_exact else [], # Store empty list if not doing exact
            'faces': faces,
            'volume': volume,
            'q': q
        }
        data.append(data_record)

        # ----------------------------------------------------- console log
        # Check if faces is an empty NumPy array
        if faces.size == 0:
            print(f'[{k+1:3d}/{n_files}] {file_info:<20s} pts={V.shape[0]:3d} NO HULL')
        else:
            # Format output similar to MATLAB
            euler_defect_str = f'{q["eulerDefect"]:+.0f}' if not np.isnan(q["eulerDefect"]) else 'NaN'
            ar_max_str = f'{q["AR_max"]:.2f}' if not np.isnan(q["AR_max"]) else 'NaN'
            print(f'[{k+1:3d}/{n_files}] {file_info:<20s} F={faces.shape[0]:3d} EulerΔ={euler_defect_str:<5s} AR_max={ar_max_str:<5s} ', end='')

            if not np.isnan(q['maxMomentErr']):
                print(f'Merr={q["maxMomentErr"]:.1e}')
            else:
                print() # Print newline

    print('✓ Done.')

    return data

try:
#     # Run the process_designs function
    design_data = process_designs(folder='/home/default/temp', tol=1e-16, do_exact=False)

#     # You can now inspect the design_data list
    print("\n--- Processed Data ---")
    for entry in design_data:
        print(f"File: {entry['file']}")
        print(f"  Pts Float Shape: {entry['ptsFloat'].shape}")
        if entry['ptsSym']:
            print(f"  Pts Sym Sample: {entry['ptsSym'][:2]}")  # Print first 2 rows
        print(f"  Faces Shape: {entry['faces'].shape if entry['faces'].size > 0 else 'Empty'}")
        print(f"  Volume: {entry['volume']}")
        print(f"  Quality Metrics: {entry['q']}")
        print("-" * 20)

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")

