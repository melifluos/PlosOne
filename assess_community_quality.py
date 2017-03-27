# !/usr/bin/env python
"""
This module assesses the quality of our ground truth communities.
We define ground-truth communities as groups of people who all share the same tag 
in our system
author: Ben Chamberlain 26/1/2014
"""

from __future__ import division
import os, csv
import numpy as np

np.seterr(all='warn')  # To show floating point errors
import cPickle, csv
from time import gmtime, strftime, sleep
from IPython import embed
import pandas as pd
import numpy.ma as ma
import networkx as nx
from itertools import combinations
from datetime import datetime
from influencer_index.src import star_index

# from science_production.utilities_python import string_to_index

# from science_production.utilities_python import string_to_index
# from science_sandbox.locality_sensitive_hashing.PR_community_detection import calculate_PR,get_cut

# Inputs
package_directory = os.path.dirname(os.path.abspath(__file__))
TAG_FILE_PATH = "resources/all_handles_tag3.csv"

# matrix_npz = os.environ['GIT_FOLDER'] + "/science_production/large_resources/community_exploration/matrix.npz"
# star_index_pickle = os.environ['GIT_FOLDER'] + "/science_production/community_exploration/resources/star_index.pickle"
# processed_matrix = "matrix_proportion.npz"
# matrix_ready = os.environ['GIT_FOLDER'] + "/science_production/large_resources/community_exploration/matrix.npz"

# load the twitter data into memory
twit_star_compression = star_index.StarCompression(network='twitter', filter='influencer')
star_lookup = twit_star_compression.lookup
max_int = twit_star_compression.max_int
twit_star_compression.load_signatures()
signatures = twit_star_compression.get_signatures()


def calculate_PR(teleport_set, A, beta=0.85, n_iterations=2):
    """
    calculate the PageRank vector
    beta - the teleport probability, in this case how much probability goes back
    to the seed at each iteration
    teleport set - a randomly chosed vertex
    A - the adjacency matrix for the community
    n_iterations - the number of iterations of PR to run, more is more accurate, 
    but in practice two is adequate
    """
    # Create seed and label strength vectors
    R = np.ones((A.shape[0], 1), dtype=np.float16)  # Label strength vector
    R_total = np.sum(R, dtype=np.float32)  # Total staring PageRank
    S = np.zeros((A.shape[0], 1), dtype=np.float16)  # Seeds vector
    S[teleport_set] = 1  # Set seeds
    S_num = S.sum()  # Number of seeds
    # propagate label with teleportation
    for dummy_i in range(n_iterations):
        R = beta * np.dot(A, R)
        missing_R = R_total - R.sum()  # Determine the missing PageRank and reinject into the seeds
        R += (missing_R / S_num) * S
    return R.flatten()


def get_cut(PR_vector, degree_vector, A):
    """
    Conducts a Sweep through the PageRank vector
    A Sweep is a commonly used method of extracting a graph cut from a vector.
    Sweeps through the PageRank vector repeatedly recalculating conductance
    and storing the minimum found
    """
    # get a degree normalized PR vector
    dn_vector = np.divide(PR_vector, degree_vector)
    # sort in descending order
    sorted_idx = np.argsort(dn_vector)[::-1]

    # TODO: ONLY NEED TO ITERATE OVER NON-ZERO PR VECTORS
    community_size = len(dn_vector)
    total_edge_volume = np.sum(A)  # This sums the edge volume attaced to every node. The double counting is deliberate
    # conduct a Sweep - a te
    min_conductance = 1.0
    for set_size in xrange(2, community_size):
        # get the set with the set_size largest PR vectors
        cut_indices = sorted_idx[0:set_size]
        # get the adjacency matrix for this subgraph
        cut_edges = A[cut_indices, :]
        if 2 * np.sum(cut_edges) > total_edge_volume:
            break  # due to the min in the denominator of conductance, no lower conductance can be found
        internal_edges = cut_edges[:, cut_indices]
        internal_weight = get_internal_weight(internal_edges, community_size)
        external_weight = get_external_weight(cut_edges, internal_weight)
        conductance = calculate_conductance(internal_weight, external_weight)
        min_conductance = min(min_conductance, conductance)
    return min_conductance


def get_community_ids(tag):
    """
    extracts all ids of stars who's tag is tag
    """
    # Load tags
    tags = pd.io.parsers.read_csv(TAG_FILE_PATH)
    tags = tags.drop_duplicates('Handle')  # Remove duplicates
    tags = tags.set_index('Handle')  # Make handle the index

    community = tags[tags['Tag'].str.contains(tag, na=False, case=False)]

    ids = community['NetworkID']
    return ids.tolist()


def create_graphml(A, communities, handles, output, threshold=0.0):
    """
    Produces a graph ml file suitable for visualising in Gephi
    writes out to the path specified by output
    not very efficient, but I can't find a vectorised way of adding attributes
    :param A: signatures matrix
    :param communities: The indices of the members of the communities. This can be either into the influencer of star matrix
    :param handles: The handles of the community members
    :param output:
    :param threshold:
    :return:
    """
    g = nx.Graph()

    community_size, n_star = A.shape

    internal_edges = A[:, communities]

    for idx, handle in enumerate(handles):
        g.add_node(idx, name=handle)
        for out_idx, edge in enumerate(internal_edges[idx, :]):
            if out_idx < idx:
                if edge > threshold:
                    g.add_edge(out_idx, idx, weight=float(edge))
                    # print float(edge)

    nx.write_graphml(g, output)


def get_jaccards(signatures, star_lookup, indices, just_stars=True):
    """
    a vectorised one versus all Jaccard calculation
    :param signatures: the matrix of minhash signatures for all influencers
    :param indices: The indices into the signatures matrix for the community members
    :param just_stars: If true, only compare with other stars rather than all influencers. This need to be true unless
    you have 32GB+ of RAM or memory will get blown
    """
    try:
        community_size = len(indices)
    except:
        community_size = 1

    n_stars, n_hashes = signatures.shape

    if just_stars:
        lookup_df = star_lookup.df
        star_lookup = lookup_df[lookup_df["isstar"] == True]
        star_indices = star_lookup.index.values
        # get rid of indices that were added after the last minhashes were generated
        star_indices = star_indices[star_indices < signatures.shape[0]]
        n_stars = len(star_indices)
        star_signatures = signatures[star_indices, :]

    print 'calculating jaccard coefficients for', community_size, 'community members against ', n_stars, ' stars'

    jaccards = np.zeros((community_size, n_stars))

    for jacc_idx, star in enumerate(indices):
        comparison_signature = signatures[star, :]

        assert comparison_signature[0] != np.iinfo(
            np.uint64).max, "attempting to calculate the Jaccard of a star that we don't have data for"

        # tile for broadcasting
        tiled_community = np.tile(comparison_signature, (n_stars, 1))
        # do a vectorize element-wise star1 == star_j for all j
        if just_stars:
            collisions = star_signatures == tiled_community
        else:
            collisions = signatures == tiled_community

        jacc = np.sum(collisions, 1) / n_hashes

        jaccards[jacc_idx, :] = jacc

        print 'community member ', jacc_idx, ' complete'

    return jaccards


def get_jaccards_vec(signatures, idx):
    """
    a very vectorised one versus all Jaccard calculation
    overflows memory for large communities though
    """
    try:
        community_size = len(idx)
    except:
        community_size = 1
    n_stars, n_hashes = signatures.shape

    comparison_signatures = signatures[idx, :]
    # reshape for broadcasting
    comparison_signatures = comparison_signatures.reshape((community_size, 1, n_hashes))
    # do a vectorize element-wise star1 == star_j for all j
    tiled_sigs = np.tile(signatures, (community_size, 1, 1))

    tiled_community = np.tile(comparison_signatures, (1, n_stars, 1))

    collisions = tiled_sigs == tiled_community

    return np.sum(collisions, 2) / n_hashes


def get_node_list(node_file, col_idx=0, has_header=True):
    """
    takes a csv of ids and returns a list of ids
    col_idx - the column in node_file containing the ids
    """
    retval = []
    with open(node_file, 'r') as f:
        reader = csv.reader(f)
        if has_header:
            reader.next()
        for line in reader:
            retval.append(line[col_idx])
    return retval


def get_internal_weight(internal_edges, community_size):
    """
    calculates the weight of internal edges
    internal_edges - the edges weights inside the community
    community_size - number of vertices in the community
    NOTE: people sometimes define internal edge weight by summing the edges of each node independantly, which 
    would give a result twice as large as this function's
    """
    m_s = np.sum(internal_edges) / 2  # matrix is symmetric, only count edges once
    return m_s


def get_external_weight(A, internal_edge_weight):
    """
    calculate the weight of external edges ie. those that 
    connect a node inside of the community with one outside
    A is the adjacency matrix for JUST the nodes in the community
    internal_edge_weight is the sum of the edge weights that connect two nodes in the community    
    """
    total_edge_weight = np.sum(A)  # This is 2*internal_edges + external_edges
    external_edge_weight = total_edge_weight - 2 * internal_edge_weight
    return external_edge_weight


def calculate_seperability(m_s, c_s):
    """
    m_s is the total weight of edges between two nodes within the community
    c_s is the total weight of edges where just one node is in the community
    """
    if c_s == 0:
        # TODO - not really sure what to do about disconnected communities, this number should be high though
        seperability = m_s
    else:
        seperability = m_s / c_s
    return seperability


def calculate_density(m_s, community_size):
    """
    calculates the density of the communities
    measures the fraction of the edges out of all possible edges 
    that the community contains
    """
    assert community_size > 1, "communities are only defined for two or more nodes"
    max_possible_edge_weight = community_size * (community_size - 1) / 2

    density = m_s / max_possible_edge_weight

    return density


def calculate_cohesiveness(jaccs, n_restarts, community_size):
    """
    calculates the cohesiveness of a community by running sub-community detection 
    and returning the conductance of the lowest conductance sub-community
    n_restarts - the number of randomly selected seeds to try community detection from 
    """
    cohesiveness = 1
    # get the weighted degree of each vertex
    degree = np.sum(jaccs, 0)
    # randomly select the vertices to start with
    starting_vertices = np.random.choice(range(community_size), n_restarts)
    # prepare the adjacency matrix for PageRank by degree normalizing
    adj_mat = np.divide(jaccs, np.tile(degree, (community_size, 1)))

    for vertex in starting_vertices:
        pr = calculate_PR(vertex, adj_mat)
        cohesiveness = min(get_cut(pr, degree, jaccs), cohesiveness)
    return cohesiveness


def calculate_conductance(m_s, c_s, total_edge_weight=float("inf")):
    """
    calculates the conductance of the communities
    m_s is the total internal edge weight (edges between two nodes within the community)
    c_s is the total external edge weight (edges where just one node is in the community)     
    the denominator is actually min(V(s),V(\s)) normally we are only interested in small subsets 
    so V(s) is always smaller, but if in doubt, enter the total edge volume
    """
    if c_s == 0 and m_s == 0:
        conductance = 1
    elif total_edge_weight > 2 * m_s:
        conductance = c_s / (2 * m_s + c_s)
    else:
        conductance = c_s / (2 * (total_edge_weight - m_s) + c_s)
    return conductance


def calculate_clustering_coefficient(internal_edges):
    """
    calculates the clustering coefficient for a weighted graph following the method of Holme et al.
    in P. Holme, S.M. Park, B.J. Kim, and C.R. Edling, Physica A 373, 821 (2007)
    """

    connected_vertices = np.sum(internal_edges, 0) != 0
    print 'removed ', internal_edges.shape[0] - np.sum(connected_vertices), ' vertices'

    internal_edges = internal_edges[connected_vertices, :][:, connected_vertices]

    # cube the matrix, probably a better way to do this
    w3 = internal_edges.dot(internal_edges).dot(internal_edges)

    w_max = np.ones(internal_edges.shape)

    denom = internal_edges.dot(w_max).dot(internal_edges)

    if np.sum(np.diag(denom) == 0) > 0:
        print 'there are ', np.sum(np.diag(denom) == 0), 'zeros in the denominator of the clustering coefficient equation \
        ABOUT TO GET DIVIDE BY ZERO ISSUES!!!'

    # get the clustering coefficients for each node
    clustering_coefficients = np.divide(np.diag(w3), np.diag(denom))

    # we characterize the community by its average clustering coefficient
    return np.mean(clustering_coefficients)


def output_analysis(out_file, tag, clustering_coefficient, cohesiveness, density, seperability):
    """
    write out the results to file
    """
    full_path = out_file + '/' + tag + '.csv'
    with open(full_path, 'wb') as f:
        writer = csv.writer(f)


def change_indices(indices, star_lookup):
    """
    switch from influencer indices to star indices
    :param indices:
    :param star_lookup: The index to id lookup table
    :return:
    """
    from copy import deepcopy
    df = star_lookup.df
    star_df = deepcopy(df[df['isstar'] == True])
    star_df['star_index'] = range(len(star_df))
    star_df['star_index'] = star_df['star_index'].astype(int)
    print star_df.head()
    indices = np.array(indices)
    star_indices = star_df.loc[indices, :]['star_index'].dropna()
    print 'number that are stars ', len(star_indices)
    print star_indices.head()
    return star_indices


def filter_none_stars(ids, star_lookup):
    """
    removes non-stars
    :param ids:
    :param star_lookup:
    :return:
    """
    df = star_lookup.df
    stars = df[df["isstar"] == True]
    star_ids = stars['id'][stars['id'].isin(ids)]
    return star_ids


def run_analysis_suite(ids, signatures, star_lookup, generate_graphml=False, tag=None):
    """ 
    runs the full suite of community analysis metrics for a single community
    """
    assert len(ids) > 1, "communities are only defined for two or more nodes"
    # set the number of restarts for cohesiveness - each restart tries to find a different sub-community
    n_iterations = 10
    # make sure they are all stars
    print 'number of ids', len(ids)
    ids = filter_none_stars(ids, star_lookup)
    print 'number of stars', len(ids)
    # get the indices into our Jaccard matrix / minhash signatures
    indices = np.array(star_lookup.id(ids)['index'])
    # remove any stars that have been added to the index since the minhashes were last updates
    indices = indices[indices < signatures.shape[0]]
    # remove any indices that we don't have stars for - these indices have all signatures set to max int
    good_stars = signatures[indices, 0] != max_int
    indices = indices[good_stars]
    # get the community size, this may not be the same as the number of ids as not all ids are indexed
    indices = list(indices)
    community_size = len(indices)
    # extract all weighted edges for the community

    jaccs = get_jaccards(signatures, star_lookup, indices)
    handles = star_lookup.index(indices)['handle']
    # switch from influencer to star indices
    indices = change_indices(indices, star_lookup)
    if generate_graphml:
        create_graphml(jaccs, indices, handles,
                       'local_results/' + tag + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '.graphml')
    # get just internal edges
    internal_edges = jaccs[:, indices]
    # remove the self loops
    np.fill_diagonal(internal_edges, 0)
    # calculate some of the parameters of the metrics
    internal_weight = get_internal_weight(internal_edges, community_size)
    external_weight = get_external_weight(jaccs, internal_weight)
    # get the four metrics
    clustering_coefficient = calculate_clustering_coefficient(internal_edges)
    cohesiveness = calculate_cohesiveness(internal_edges, n_iterations, community_size)
    density = calculate_density(internal_weight, community_size)
    seperability = calculate_seperability(internal_weight, external_weight)
    conductance = calculate_conductance(internal_weight, external_weight)
    # the ratio of external to internal conductance
    conductance_ratio = conductance / cohesiveness
    # return in alphabetic order
    return [community_size, clustering_coefficient, cohesiveness, conductance, conductance_ratio, density, seperability]


def check_clustering(A, node):
    w3 = internal_edges.dot(internal_edges).dot(internal_edges)
    print w3
    w_max = np.ones(internal_edges.shape)
    print w_max
    denom = internal_edges.dot(w_max).dot(internal_edges)
    print denom
    clustering_coefficients = np.divide(np.diag(w3), np.diag(denom))
    return clustering_coefficients


if __name__ == '__main__':

    full_path = 'local_results/community_analysis' + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv'
    header = ['community', 'size', 'clustering_coefficient', 'cohesiveness', 'conductance', 'conductance_ratio',
              'density', 'separability']
    with open(full_path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    with open('resources/tags_to_plot.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            tag = line[0]
            if tag.lower() == 'community':
                continue  # file has a header
            elif tag.lower() == 'mixed martial arts':
                tag = 'taekwondo'
            elif tag.lower() == 'adult actor':
                tag = 'pornstar'
            ids = get_community_ids(tag)
            print tag, ' has ', len(ids), ' members'
            results = run_analysis_suite(ids, signatures, star_lookup, generate_graphml=True, tag=tag)
            print header
            print 'results are ', results
            with open(full_path, 'ab') as f:
                writer = csv.writer(f)
                output = [tag] + results
                writer.writerow(output)

    """
    full_path = './community_analysis.csv'
    with open(full_path,'a') as f:
        writer = csv.writer(f)
        writer.writerow(['community',' size','clustering_coefficient','cohesiveness','conductance','conductance_ratio','density','separability'])
    
        athletics_path = 'D:/workspace/science_sandbox/locality_sensitive_hashing/local_resources/ICWSM15/athletics/athletics_handles.csv'
        tag = 'athletics'
        ids = get_node_list(athletics_path,0)
        ids = ids[0:100]
        results = run_analysis_suite(ids,signatures,False,tag)
        output = [tag] + results
        writer.writerow(output)
        print results
#    conductance = calculate_conductance(signatures,indices)
#    print conductance

# test with adidas and nike
   # jaccs = get_jaccards(signatures,[207,10035])
    #print a.shape
    #print a[:,[207,10035]]
    
    # test on some fully connected binary networks
    internal_edges = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]],dtype=float)
    #internal_edges = np.array([[0,1,1,1,1],[1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]],dtype=float)  
    #internal_edges = np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=float)
    n_iterations = 3
    clustering = calculate_clustering_coefficient(internal_edges)
    print 'clustering, ', clustering
    cohesiveness = calculate_cohesiveness(internal_edges,n_iterations,internal_edges.shape[0])
    print cohesiveness
      
    print 'CHEKCING FUNCTION'
    print check_clustering(internal_edges,1)
    """
