"""Module containing strategies to assign tutors to students
"""

import copy
import collections
import warnings
import numpy as np
from ortools.graph import pywrapgraph
#import pdb; pdb.set_trace()


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




def dichotomySearchMaxMin(n_students_per_tutor, Affinity, low_val=0, refine_flag = True):
    
    T, S = Affinity.shape
    
    affinity_sorted = np.unique(np.reshape(Affinity[Affinity > 0],(1,-1)))
    #affinity_sorted = np.sort(affinity_sorted)
    upp_ind = len(affinity_sorted) - 1
  
    if low_val == 0:
        low_ind = 0
    else:
        ii = np.where(affinity_sorted > low_val)[0]
        if len(ii) == 0:
            low_ind = 0
        else:
            low_ind = ii[0] - 1
    
    # check whether lower problem is feasible 
    feasible_flag, unassigned_students, min_affinity_tmp, tutor_student_assignment_tmp, tutor_student_assignment_tmp_1 = \
        MaxFlow_FeasibilityProblem(n_students_per_tutor, Affinity, affinity_sorted[0])
        
    if feasible_flag==False:  
        return feasible_flag, unassigned_students, min_affinity_tmp, tutor_student_assignment_tmp
    else:
        min_affinity = min_affinity_tmp
        tutor_student_assignment = tutor_student_assignment_tmp
        tutor_student_assignment_1 = tutor_student_assignment_tmp_1
        
    # check whether upper problem is feasible (normally it should be unfeasible) 
    feasible_flag, unassigned_students_tmp, min_affinity_tmp, tutor_student_assignment_tmp, tutor_student_assignment_tmp_1 = \
        MaxFlow_FeasibilityProblem(n_students_per_tutor, Affinity, affinity_sorted[upp_ind])
        
    if feasible_flag:
        return feasible_flag, [], min_affinity_tmp, tutor_student_assignment_tmp

    # dichotomic search
    while (upp_ind>low_ind+1):
        
        new_ind = int((low_ind + upp_ind)/2)
        
        feasible_flag, unassigned_students, min_affinity_tmp, tutor_student_assignment_tmp, tutor_student_assignment_tmp_1 = \
        MaxFlow_FeasibilityProblem(n_students_per_tutor, Affinity, affinity_sorted[new_ind])

        if feasible_flag:
            low_ind = np.where(affinity_sorted==min_affinity_tmp)[0][0] #new_ind
            min_affinity =  min_affinity_tmp #affinity_sorted[low_ind]
            tutor_student_assignment = tutor_student_assignment_tmp
            tutor_student_assignment_1 = tutor_student_assignment_tmp_1
        else:
            upp_ind = new_ind
    
    feasible_flag = True
    unassigned_students = []
    
    if refine_flag & (len(tutor_student_assignment[0])>1):
        #import pdb; pdb.set_trace()
        tutor_student_assignment = refine_solution_Transportation(Affinity, n_students_per_tutor, min_affinity)
    
    
    return feasible_flag, unassigned_students, min_affinity, tutor_student_assignment



def MaxMinStudentTutorAssignment(Affinity, n_students_per_tutor, refine_flag = 1):
    
    T, S = Affinity.shape
    n_students_per_tutor = np.array(n_students_per_tutor)
    
    S_set0 = np.array(range(S))
    S_set2 = np.array([])
    final_assignment = [[] for ii in range(S)] # ...[ss] = set of tutors assigned to student ss and corresponding affinity 

    T_set = np.where(n_students_per_tutor>0)[0] # set of tutors
    n_students_per_tutor = n_students_per_tutor[T_set]

    while (len(T_set)>0) & (np.sum(Affinity)>0):
        
        S_set1 = np.setdiff1d(S_set0, S_set2)
        min_affinity = 0

        # Assign at most one tutor to each student according to a max-min allocation
        while (len(S_set1)>0) & (len(T_set)>0):

            # find the max-min value by dichotomy search
            Affinity_1 = Affinity[T_set][:,S_set1]
            
            feasible_flag, unassigned_students, min_affinity, tutor_student_assignment = \
                dichotomySearchMaxMin(n_students_per_tutor, Affinity_1, min_affinity, refine_flag)
            
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



def refine_solution_Transportation(Affinity, n_students_per_tutor, min_affinity):
    
    T, S = Affinity.shape
    
    # build the min-cost transportation problem
    # nodes and edges
    ind = np.where(Affinity>=min_affinity)
    start_nodes = ind[0] # tutors
    end_nodes = ind[1] + T # students

    # add artificial node to make sum(supplies)=0
    start_nodes = np.r_[start_nodes, range(T)] 
    end_nodes = np.r_[end_nodes, [S+T]*T] # students

    n_edges = len(start_nodes)

    # cost vector, capacities and supplies
    cost_vec = np.zeros(len(ind[0]))
    cost_vec[Affinity[ind]==min_affinity] = 1
    cost_vec = np.r_[cost_vec, np.zeros(T)]
    
    capacities = [np.max(n_students_per_tutor)] * n_edges

    supplies = np.r_[n_students_per_tutor, [-1] * S, S-np.sum(n_students_per_tutor)]

    # Instantiate a SimpleMinCostFlow solver
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc
    for i in range(0,len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(int(start_nodes[i]), \
                                int(end_nodes[i]), int(capacities[i]), int(cost_vec[i]))

    # Add node supplies
    for i in range(0,len(supplies)):
        min_cost_flow.SetNodeSupply(i, int(supplies[i]))
    
    # check that it is feasible
    if min_cost_flow.Solve() != min_cost_flow.OPTIMAL:
        print('There was an issue with the min-cost flow input.')
    
    keep_it = [False for ii in range(n_edges)]
    for ii in range(n_edges):
        keep_it[ii] = (min_cost_flow.Flow(ii)==1) & (min_cost_flow.UnitCost(ii)==1)
    
    tutor_student_assignment = [start_nodes[keep_it], end_nodes[keep_it]-T]
    
    return tutor_student_assignment



def MaxFlow_FeasibilityProblem(n_students_per_tutor, Affinity, min_val):
    
    T, S = Affinity.shape
    
    # build the min-cost transportation problem
    # nodes and edges
    ind = np.where(Affinity>=min_val)
    start_nodes = ind[0] # tutors
    end_nodes = ind[1] + T # students
    n_edges_inner = len(start_nodes)
    
    # add source [S+T] -> tutors and students -> destination [S+T+1]
    start_nodes = np.r_[start_nodes, [S+T] * T, T+np.arange(S)]
    end_nodes = np.r_[end_nodes, range(T), [S+T+1] * S]
    
    # capacities
    capacities = np.r_[np.ones(n_edges_inner), n_students_per_tutor, np.ones(S)]
    
    max_flow = pywrapgraph.SimpleMaxFlow()
    
    # Add each arc
    for ii in range(len(start_nodes)):
        max_flow.AddArcWithCapacity(int(start_nodes[ii]), int(end_nodes[ii]), int(capacities[ii]))
    
    if max_flow.Solve(S+T, S+T+1) != max_flow.OPTIMAL:
        print('There was an issue with the max-flow input.')
    
    # compute the tutor-student resulting allocation
    allocation = np.zeros([T,S], dtype=int)
    for ii in range(n_edges_inner):
        allocation[start_nodes[ii], end_nodes[ii]-T] = int(max_flow.Flow(ii))
    
    # students without a tutor
    unassigned_students = np.where(np.sum(allocation,axis=0)==0)[0]
    feasible_flag = (len(unassigned_students)==0)
   
    # minimum achieved affinity
    tmp = allocation * Affinity
    if feasible_flag==False:
        min_affinity = 0
    else:
        min_affinity = np.min(tmp[tmp>0])
    
    if np.sum(tmp)==0:
        tutor_student_assignment_low = []
        tutor_student_assignment_high = []
    else:
        if feasible_flag==False:
            tutor_student_assignment_low = []
            tutor_student_assignment_high = np.where(tmp>min_affinity)
        else:
            tutor_student_assignment_low = np.where(tmp==min_affinity)
            tutor_student_assignment_high = np.where(tmp>min_affinity)
   
    return feasible_flag, unassigned_students, min_affinity, tutor_student_assignment_low, tutor_student_assignment_high

