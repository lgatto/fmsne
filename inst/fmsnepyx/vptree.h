/*
 *  vptree.h
 *  Implementation of a vantage-point tree.
 *
 *  Authors: this code was created by Laurens van der Maaten, and modified by Cyril de Bodt (ICTEAM - UCLouvain).
 *  @email: cyril __dot__ debodt __at__ uclouvain.be
 *  Last modification date: May 30th, 2020
 *  The original version of the code is available at: https://github.com/lvdmaaten/bhtsne/blob/master/vptree.h (last consulted on May 27, 2020).
 *  
 *  Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 *  All rights reserved.
 *
 *  You can use, modify and redistribute this software freely, but not for commercial purposes. 
 *  The use of this software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *  3. All advertising materials mentioning features or use of this software
 *     must display the following acknowledgement:
 *     This product includes software developed by the Delft University of Technology.
 *  4. Neither the name of the Delft University of Technology nor the names of 
 *     its contributors may be used to endorse or promote products derived from 
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 *  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
 *  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
 *  EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
 *  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
 *  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
 *  OF SUCH DAMAGE.
 *
 */


#include <stdlib.h>   // for malloc, rand, etc.
#include <algorithm>  // for nth_element
#include <stdio.h>    // for fprintf
#include <queue>      // for priority_queue
#include <cfloat>     // for DBL_MAX
#include <cmath>      // for sqrt

#ifndef VPTREE_H
#define VPTREE_H

double euclidean_distance_datapoint(const double* x1, const double* x2, int dim) {  // Euclidean distance between two data points 
    double diff = x1[0] - x2[0];
    double dd = diff * diff;
    int d;
    for (d=1; d<dim; d=d+1) {
        diff = x1[d] - x2[d];
        dd += diff * diff;
    }
    return sqrt(dd);
}

class VpTree {
public:
    // Constructor create a new VpTree from data. The data are stored in a one-dimensional array. _D must be at least 1.
    VpTree(const double* X, int N, int D) {
        _X = X;
        _D = D;
        _idx = (int*) malloc(N * sizeof(int));
        if (_idx == NULL) {
            fprintf(stderr,"Out of memory.\n");
            exit(EXIT_FAILURE);
        }
        int n;
        for (n=0; n<N; n=n+1) {
            _idx[n] = n;
        }
        // An array storing the distances between a vantage-point candidate and a random sample of the data points
        _dsamp = (double*) malloc( ((int) sqrt(N) + 1) * sizeof(double));
        if (_dsamp == NULL) {
            free(_idx);
            fprintf(stderr,"Out of memory.\n");
            exit(EXIT_FAILURE);
        }
        // Construct the tree from the root
        _root = buildFromPoints(0, N);
        // Free _idx and _dsamp as they are not useful anymore. They were both used in buildFromPoints.
        free(_idx);
        free(_dsamp);
    }
    
    // Destructor. It frees the ressources allocated in the object and the object. 
    ~VpTree() {
        delete _root;
    }
    
    // Function that uses the tree to find the k nearest neighbors of x
    void search(const double* x, int k, int* indices) {
        // Use a priority queue to store intermediate results
        std::priority_queue<HeapItem> heap;
        // Variable that tracks the distance to the farthest neighbor currently considered
        _tau = DBL_MAX;
        // Perform the search
        search_rec(_root, x, k, heap);
        // Gather the results of the heap, in reverse order due to the way the heap stores the results
        int nn;
        for (nn=k-1; nn>=0; nn=nn-1) {
            indices[nn] = heap.top().index;
            heap.pop();
        }
    }
    
private:
    const double* _X;     // Data, stored in a one-dimensional array
    int _D;         // Dimension
    int* _idx;      // Array with the indexes of the data points. Only initialized and used when building the tree. 
    double* _dsamp; // Array used to select the vantage-points. Only initialized and used when building the tree. 
    double _tau;    // Threshold for the search
    
    // Single node of a VP tree (it has a vantage point and a radius; left children contains points that are closer to the vantage point than the radius)
    struct Node {
        int index;                      // index of a point in the data set
        double threshold;               // radius
        double lmd;                     // Minimum distance to the current node in the left child
        double lMd;                     // Maximum distance to the current node in the left child
        double rmd;                     // Minimum distance to the current node in the right child
        double rMd;                     // Maximum distance to the current node in the right child
        Node* left;                     // points closer by than threshold
        Node* right;                    // points farther away than threshold
        
        Node (int idx, double left_md, double left_Md, double right_md, double right_Md, Node* le, Node* ri) {       // Constructor, initializing the fields
            index = idx;
            lmd = left_md;
            lMd = left_Md;
            rmd = right_md;
            rMd = right_Md;
            threshold = (lMd+rmd)*0.5;
            left = le;
            right = ri;
        }
        
        ~Node() {                       // Destructor
            delete left;
            delete right;
        }
    }* _root;
    
    double euclidean_distance(int i1, int i2) {       // Euclidean distance between two data points
        int idx1 = i1*_D;
        int idx2 = i2*_D;
        double diff = _X[idx1] - _X[idx2];
        double dd = diff * diff;
        int d;
        for (d=1; d<_D; d=d+1) {
            idx1 += 1;
            idx2 += 1;
            diff = _X[idx1] - _X[idx2];
            dd += diff * diff;
        }
        return sqrt(dd);
    }
    
    double euclidean_distance_target(int i1, const double* x) {       // Euclidean distance between a data point and the target
        int idx1 = i1*_D;
        double diff = _X[idx1] - x[0];
        double dd = diff * diff;
        int d;
        for (d=1; d<_D; d=d+1) {
            idx1 += 1;
            diff = _X[idx1] - x[d];
            dd += diff * diff;
        }
        return sqrt(dd);
    }
    
    // An item on the intermediate result queue
    struct HeapItem {
        int index;
        double dist;
        
        HeapItem (int ind, double d) {
            index = ind;
            dist = d;
        }
        bool operator<(const HeapItem& o) const {
            return dist < o.dist;
        }
    };
    
    // Distance comparator to use in std::nth_element
    struct DistanceComparator {
        int dim;
        const double* x;
        const double* X;
        DistanceComparator(int D, const double* x_cur, const double* X_c) {
            dim = D;
            x = x_cur;
            X = X_c;
        }
        bool operator()(int idx1, int idx2) {
            return euclidean_distance_datapoint(x, &X[idx1*dim], dim) < euclidean_distance_datapoint(x, &X[idx2*dim], dim);
        }
    };
    
    double eval_vp_cand(int ivp, int n_samp, int med, int ndp, int lower, int lower_up_vp_cand) {
        int i;
        int iidx;
        int ss;
        int last;
        int pos_best_vp = lower_up_vp_cand;
        for (ss=0; ss<n_samp; ss=ss+1) {
            i = rand()%ndp + lower;
            iidx = _idx[i];
            // Swapping to enable random sampling without replacement.
            if (i > lower_up_vp_cand) {
                ndp -= 1;
                last = lower + ndp;
                _idx[i] = _idx[last];
                _idx[last] = iidx;
            } else {
                if (i == lower_up_vp_cand) {
                    pos_best_vp = lower;
                }
                _idx[i] = _idx[lower];
                _idx[lower] = iidx;
                lower += 1;
                ndp -= 1;
            }
            // Storing the distance between iidx and ivp
            _dsamp[ss] = euclidean_distance(ivp, iidx);
        }
        
        // Putting the best vantage-point found so far back to lower_up_vp_cand
        if (pos_best_vp != lower_up_vp_cand) {
            last = _idx[pos_best_vp];
            _idx[pos_best_vp] = _idx[lower_up_vp_cand];
            _idx[lower_up_vp_cand] = last;
        }
        
        // Computing the median of the distances in _dsamp
        std::nth_element(_dsamp, _dsamp+med, _dsamp+n_samp);
        // Computing the score
        double dmed = _dsamp[med];
        double diff;
        double score = 0.0;
        for (ss=0; ss<n_samp; ss=ss+1) {
            diff = _dsamp[ss] - dmed;
            score += diff*diff;
        }
        return score;
    }
    
    void select_vp(int lower, int ndp) {
        if (ndp == 2) {
            if (rand()%2 == 1) {        // Moving the selected vantage-point at lower
                int tmp = _idx[lower];
                int lower_1 = lower+1;
                _idx[lower] = _idx[lower_1];
                _idx[lower_1] = tmp;
            }
        } else {
            // Number of candidate vantage-point to consider
            int n_samp_1 = (int) sqrt(ndp);
            int n_samp = n_samp_1+1;
            int med = n_samp/2;
            // Best vantage-point currently found and the index of the data point it refers to
            int i = rand()%ndp + lower;
            int ivp = _idx[i];
            // Swapping indexes to enable random sampling without replacement
            if (i != lower) {
                _idx[i] = _idx[lower];
                _idx[lower] = ivp;
            }
            // Updating ndp and saving the position after lower
            int lower_up = lower+1;
            ndp -= 1;
            // Score of ivp as vantage-point
            double best_score = eval_vp_cand(ivp, n_samp, med, ndp, lower_up, lower);
            // Considering n_samp_1 other vantage-point candidates
            int j;
            double score;
            int ndp_vp_cand = ndp;
            int lower_up_vp_cand = lower_up;
            for (j=0; j<n_samp_1; j=j+1){
                // Sampling another vantage-point candidate
                i = rand()%ndp_vp_cand + lower_up_vp_cand;
                ivp = _idx[i];
                // Swapping with current best vp. We need to proceed like this to use eval_vp_cand. 
                if (i != lower_up_vp_cand) {
                    _idx[i] = _idx[lower_up_vp_cand];
                }
                _idx[lower_up_vp_cand] = _idx[lower];
                _idx[lower] = ivp;
                // Computing the score of the new candidate
                score = eval_vp_cand(ivp, n_samp, med, ndp, lower_up, lower_up_vp_cand);
                // If the score is smaller than the best score found so far, putting the best vantage-point back at lower
                if (score < best_score) {
                    _idx[lower] = _idx[lower_up_vp_cand];
                    _idx[lower_up_vp_cand] = ivp;
                } else {
                    best_score = score;
                }
                // Updating lower_up_vp_cand and ndp_vp_cand
                lower_up_vp_cand += 1;
                ndp_vp_cand -= 1;
            }
        }
    }
    
    // Function that (recursively) fills the tree
    Node* buildFromPoints(int lower, int upper) {
        int dul = upper - lower;     // Difference between upper and lower
        if (dul > 1) {          // We did not arrived at a leaf yet. 
            // ==== Heuristic selection of the vantage point
            // Select the index of the vantage-point and move it at lower position
            select_vp(lower, dul);
            // Swapping lower and the vantage point
            int vpind = _idx[lower];
            
            // ==== Random selection of the vantage-point
            //int ivp_r = rand()%dul + lower;
            //int vpind = _idx[ivp_r];
            //_idx[ivp_r] = _idx[lower];
            //_idx[lower] = vpind;
            
            // Index of the data with the median distance to the selected vantage-point
            int lower_1 = lower+1;
            int median = (upper + lower_1) / 2;
            // Partition around the median
            std::nth_element(_idx + lower_1, _idx + median, _idx + upper, DistanceComparator(_D, &_X[vpind*_D], _X));
            // Computing the minimum and maximum distances in the left child
            int n_child = median - lower_1;
            double lmd;
            double lMd;
            int start;
            double v1;
            double v2;
            int i;
            if (n_child == 0) {
                lmd = 0.0; 
                lMd = 0.0;
            } else {
                if (n_child%2 == 0) {
                    v1 = euclidean_distance(vpind, _idx[lower_1]);
                    v2 = euclidean_distance(vpind, _idx[lower_1+1]);
                    if (v1<v2) {
                        lmd = v1;
                        lMd = v2;
                    } else {
                        lmd = v2;
                        lMd = v1;
                    }
                    start = lower_1+2;
                } else {
                    lmd = euclidean_distance(vpind, _idx[lower_1]);
                    lMd = lmd;
                    start = lower_1+1;
                }
                for (i=start; i<median; i=i+2) {
                    v1 = euclidean_distance(vpind, _idx[i]);
                    v2 = euclidean_distance(vpind, _idx[i+1]);
                    if (v1<v2) {
                        if (v1 < lmd) {
                            lmd = v1;
                        }
                        if (v2 > lMd) {
                            lMd = v2;
                        }
                    } else {
                        if (v2 < lmd) {
                            lmd = v2;
                        }
                        if (v1 > lMd) {
                            lMd = v1;
                        }
                    }
                }
            }
            // Computing the minimum and maximum distances in the right child
            n_child = upper - median;           // Always at least 1 data point in the right child.
            double rmd;
            double rMd;
            if (n_child%2 == 0) {
                v1 = euclidean_distance(vpind, _idx[median]);
                v2 = euclidean_distance(vpind, _idx[median+1]);
                if (v1<v2) {
                    rmd = v1;
                    rMd = v2;
                } else {
                    rmd = v2;
                    rMd = v1;
                }
                start = median+2;
            } else {
                rmd = euclidean_distance(vpind, _idx[median]);
                rMd = rmd;
                start = median+1;
            }
            for (i=start; i<upper; i=i+2) {
                v1 = euclidean_distance(vpind, _idx[i]);
                v2 = euclidean_distance(vpind, _idx[i+1]);
                if (v1<v2) {
                    if (v1 < rmd) {
                        rmd = v1;
                    }
                    if (v2 > rMd) {
                        rMd = v2;
                    }
                } else {
                    if (v2 < rmd) {
                        rmd = v2;
                    }
                    if (v1 > rMd) {
                        rMd = v1;
                    }
                }
            }
            // Recursively build the tree
            Node* left = buildFromPoints(lower_1, median);
            Node* right = buildFromPoints(median, upper);
            // Returning the new node. The treshold of the new node will be the distance to the median.
            return new Node(vpind, lmd, lMd, rmd, rMd, left, right);
        } else if (dul == 1) {    // Last data point to consider: we arrived at a leaf. 
            return new Node(_idx[lower], 0.0, 0.0, 0.0, 0.0, NULL, NULL);
        } else {                  // No more data point to consider: we arrived at a leaf.
            return NULL;
        }
    }
    
    // Helper function that searches the tree
    void search_rec(const Node* node, const double* x, int k, std::priority_queue<HeapItem>& heap) {
        // Compute distance between x and current node
        double dist = euclidean_distance_target(node->index, x);
        // If current node within radius tau
        if (dist < _tau) {
            bool full = heap.size() == k;
            // If we already collected k neighbors, remove the farthest one. 
            if (full) {
                heap.pop();
            }
            // Add current node to result list
            heap.push(HeapItem(node->index, dist));
            // Update value of tau. If full is False, we still need to test heap.size() == k, if the heap became full meanwhile. 
            if (full || heap.size() == k) {
                _tau = heap.top().dist;
            }
        }
        bool has_left = node->left != NULL;
        bool has_right = node->right != NULL;
        // If the current node has at least one child
        if (has_left || has_right) {
            // If x lies within the radius of ball
            if (dist < node->threshold) {
                // If there can still be neighbors inside the ball, recursively search left child first
                if (has_left && (dist > node->lmd - _tau) && (dist < node->lMd + _tau)) {
                    search_rec(node->left, x, k, heap);
                }

                // if there can still be neighbors outside the ball, recursively search right child
                if (has_right && (dist > node->rmd - _tau)) {
                    search_rec(node->right, x, k, heap);
                }
            // If x lies outside the radius of the ball
            } else {
                // if there can still be neighbors outside the ball, recursively search right child first
                if (has_right && (dist > node->rmd - _tau) && (dist < node->rMd + _tau)) {
                    search_rec(node->right, x, k, heap);
                }
                
                // if there can still be neighbors inside the ball, recursively search left child
                if (has_left && (dist < node->lMd + _tau)) {
                    search_rec(node->left, x, k, heap);
                }
            }
        }
    }
};

#endif
