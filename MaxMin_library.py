"""Module containing strategies to assign tutors to students
"""
import copy
import collections
import warnings
import numpy as np
#import pdb; pdb.set_trace()

# This class represents a directed graph using adjacency matrix representation
class Graph:
 
    def __init__(self,graph):
        self.graph = graph # residual graph
        self.ROW = len(graph)
 
    def BFS(self,s, t, parent):
        '''Returns true if there is a path from source 's' to sink 't' in
        residual graph. Also fills parent[] to store the path '''

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)
       
        # Create a queue for BFS
        queue = collections.deque()
        
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True
        
        # Standard BFS Loop
        while queue:
            u = queue.popleft()
        
            # Get all adjacent vertices's of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return visited[t]
            
    # Returns the maximum flow from s to t in the given graph
    def EdmondsKarp(self, source, sink):
        # network capacity
        capacity = copy.deepcopy(self.graph)
       
        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0 # There is no flow initially
       
        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
          
        flow = np.array([[0. for ii in range(self.ROW)] for ii in range(self.ROW)])
        for i1 in range(self.ROW):
            for i2 in range(self.ROW):
                flow[i1][i2] = np.max([capacity[i1][i2] - self.graph[i1][i2], 0])
 
        return max_flow, flow


  
def generateAffinity(S, T, n_subjects = 5, n_different_values = None):

    student_interests = np.zeros((S,n_subjects))
    student_feature = np.random.uniform(low=.5,high=1,size=S)
    for tt in range(S):
        # vec = np.random.normal(scale=student_feature[tt],size=(1,n_subjects))
        vec = student_feature[tt]*np.random.uniform(low=-1,high=1,size=(1,n_subjects))
        student_interests[tt,] = np.maximum(vec,np.zeros((1,n_subjects)))

    tutor_specialties = np.zeros((T,n_subjects))
    tutor_feature = np.random.uniform(low=.8,high=1,size=T)
    for tt in range(T):
        # vec = np.random.normal(scale=tutor_feature[tt],size=(1,n_subjects))
        vec = tutor_feature[tt]*np.random.uniform(low=-1,high=1,size=(1,n_subjects))
        tutor_specialties[tt,] = np.maximum(vec,np.zeros((1,n_subjects)))

    Affinity = np.matmul(tutor_specialties, student_interests.T)
    Affinity = np.round(Affinity*100)
   
    if (n_different_values != None):
        max_val = float(np.max(Affinity))
        for kk in range(n_different_values):
            Affinity[(Affinity<=(kk+1)*max_val/n_different_values) & \
                    (Affinity>kk*max_val/n_different_values)] = kk + 1
   
    return Affinity # T x S



def check_input_consistency(S,T,n_students_per_tutor,n_subjects):
    
    if np.sum(n_students_per_tutor)<S:
        warnings.warn("not all students can be tutored")
        return -1
    return 0



def MaxFlow_FeasibilityProblem(S, T, n_students_per_tutor, Affinity):
   
    # compute the max-flow on the modified graph
    Adj = Affinity>0
    A = np.zeros((S+T+2,S+T+2))
    source = S+T
    destination = S+T+1
    A[S:S+T,:S] = Adj
    A[:S,destination] = 1
    A[source,S:S+T] = n_students_per_tutor.T

    G = Graph(A)
    max_flow, flow = G.EdmondsKarp(source,destination)
   
    # compute the tutor-student resulting allocation
    allocation = flow[S:S+T,:S].astype(int)
   
    # students without a tutor
    unassigned_students = np.where(np.sum(allocation,axis=0)==0)[0]
   
    # minimum achieved affinity
    tmp = allocation * Affinity
    min_affinity = np.min(tmp[tmp>0])
    feasible_flag = (len(unassigned_students)==0)
    tutor_student_assignment = np.where(tmp==min_affinity)
   
    #import pdb; pdb.set_trace()
   
    return feasible_flag, unassigned_students, min_affinity, tutor_student_assignment




def dichotomySearchMaxMin(S, T, n_students_per_tutor, Affinity, low_val=0):
   
    affinity_sorted = np.unique(np.reshape(Affinity[Affinity > 0],(1,-1)))
    #affinity_sorted = np.sort(affinity_sorted)
    upp_ind = len(affinity_sorted) - 1
  
    if low_val == 0:
        low_ind = 0
    else:
        #import pdb; pdb.set_trace()
        ii = np.where(affinity_sorted > low_val)[0]
        if len(ii) == 0:
            low_ind = 0
        else:
            low_ind = ii[0] - 1
    
    # check whether lower problem is feasible 
    feasible_flag, unassigned_students, min_affinity_tmp, tutor_student_assignment_tmp = \
        MaxFlow_FeasibilityProblem(S, T, n_students_per_tutor, Affinity)
    
    if feasible_flag==False:
        return feasible_flag, unassigned_students, min_affinity_tmp, tutor_student_assignment_tmp
    else:
        min_affinity = min_affinity_tmp
        tutor_student_assignment = tutor_student_assignment_tmp
    
    # check whether upper problem is feasible (normally it should be unfeasible) 
    Affinity_tmp = np.zeros(Affinity.shape)
    Affinity_tmp[:] = Affinity
    Affinity_tmp[(Affinity_tmp<affinity_sorted[upp_ind]).reshape(Affinity.shape)] = 0

    feasible_flag, unassigned_students_tmp, min_affinity_tmp, tutor_student_assignment_tmp = \
        MaxFlow_FeasibilityProblem(S, T, n_students_per_tutor, Affinity_tmp)
    
    if feasible_flag:
        return feasible_flag, [], min_affinity_tmp, tutor_student_assignment_tmp

    # dichotomic search
    while (upp_ind>low_ind+1):
        
        new_ind = int((low_ind + upp_ind)/2)

        Affinity_tmp = np.zeros(Affinity.shape)
        Affinity_tmp[:] = Affinity
        Affinity_tmp[(Affinity_tmp<affinity_sorted[new_ind]).reshape(Affinity.shape)] = 0

        feasible_flag, unassigned_students, min_affinity_tmp, tutor_student_assignment_tmp = \
            MaxFlow_FeasibilityProblem(S, T, n_students_per_tutor, Affinity_tmp)

        if feasible_flag:
            low_ind = np.where(affinity_sorted==min_affinity_tmp)[0][0] #new_ind
            min_affinity =  min_affinity_tmp #affinity_sorted[low_ind]
            tutor_student_assignment = tutor_student_assignment_tmp
        else:
            upp_ind = new_ind
    
    feasible_flag = True
    unassigned_students = []
    
    
    return feasible_flag, unassigned_students, min_affinity, tutor_student_assignment




def MaxMinStudentTutorAssignment(Affinity, n_students_per_tutor, S=None, T=None):
    
    if (S is None) | (T is None):
        T, S = Affinity.shape
    
    S_set0 = np.array(range(S))
    S_set2 = np.array([])
    final_assignment = [[] for ii in range(S)] # ...[ss] = set of tutors assigned to student ss and corresponding affinity 

    T_set = np.where(n_students_per_tutor>0)[0] # set of tutors
    n_students_per_tutor = n_students_per_tutor[T_set]

    while (len(T_set)>0):
        
        S_set1 = np.setdiff1d(S_set0, S_set2)
        min_affinity = 0

        # Assign at most one tutor to each student according to a max-min allocation
        while (len(S_set1)>0) & (len(T_set)>0):

            # find the max-min value by dichotomy search
            Affinity_1 = Affinity[T_set][:,S_set1]
            feasible_flag, unassigned_students, min_affinity, tutor_student_assignment = \
                dichotomySearchMaxMin(len(S_set1), len(T_set), n_students_per_tutor, Affinity_1, min_affinity)

            if feasible_flag is False:
                # update the set of available students
                S_set2 = np.r_[S_set2, S_set1[unassigned_students]]
                S_set1 = np.delete(S_set1, unassigned_students)

            else:
                # update the Affinity matrix
                for kk in range(len(tutor_student_assignment[0])):
                    Affinity[T_set[tutor_student_assignment[0][kk]],S_set1[tutor_student_assignment[1]][kk]] = 0

                # update the assignment
                for kk in range(len(tutor_student_assignment[1])):
                    stud = S_set1[tutor_student_assignment[1][kk]]
                    tut = T_set[tutor_student_assignment[0][kk]]
                    final_assignment[stud].append((tut, min_affinity)) 
                    #np.append(final_assignment[stud], (tut, min_affinity))

                # update the number of students per tutor
                tut_tmp = np.unique(tutor_student_assignment[0])
                for tut in tut_tmp:
                    n_students_per_tutor[tut] -= np.sum(tutor_student_assignment[0]==tut)
                ind0 = np.where(n_students_per_tutor==0)[0]
                n_students_per_tutor = np.delete(n_students_per_tutor, ind0)

                # update the set of tutors
                T_set = np.delete(T_set, ind0)

                # update the set of students yet to be assigned
                S_set1 = np.delete(S_set1, tutor_student_assignment[1])
            
    return final_assignment

