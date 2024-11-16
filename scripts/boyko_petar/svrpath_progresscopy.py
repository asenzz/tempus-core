import numpy as np
import scipy.linalg
import scipy.optimize
import time
from six.moves import cPickle as pickle

fileNameToLoad = "progress_state_at"
res = { 'alphas' : [], 'gammas' : [], 'beta0' : [], 'maes' : [], 'errors' : [], 
       'eps' : [], 'Lambda':[], 'ElbowLeft' : [], 'ElbowRight' : [], 'Center' : [], 'RightRegion' : [], 'LeftRegion' : [], 
       'TimeTaken' : float('Inf'), 'fl' : [],     'LastEventPoint' : None, 'LastEvent' : None }
globaltype = np.float64

def svrload(K, y, Lambda):
    #if DebugPrint:
    #    print('\nsvrload')
    resultsMatrixToLoad = pickle.load(open(fileNameToLoad+'.pickle','rb'))
    alphasAll = resultsMatrixToLoad['alphas']
    numRows = len(alphasAll)
    numRows = numRows - 1 
    #TODO Choose which row to pick; instead of picking latest row
    alphas = alphasAll[numRows]
    eps =  resultsMatrixToLoad['eps'][numRows]
    ElbowLeft =  resultsMatrixToLoad['ElbowLeft'][numRows]
    ElbowRight = resultsMatrixToLoad['ElbowRight'][numRows]
    Center =  resultsMatrixToLoad['Center'][numRows]
    LeftRegion =  resultsMatrixToLoad['LeftRegion'][numRows]
    RightRegion = resultsMatrixToLoad['RightRegion'][numRows]
    gammas =  resultsMatrixToLoad['gammas'][numRows]
    beta0 =  resultsMatrixToLoad['beta0'][numRows]
    mae = resultsMatrixToLoad['maes'][numRows]       
    fl = resultsMatrixToLoad['fl'][numRows]  
    LastEventPoint = resultsMatrixToLoad['LastEventPoint']  
    LastEvent = resultsMatrixToLoad['LastEvent']
     
#    Kstar = np.zeros(shape=[1,1], DTYPE=globaltype)
#    Kstar = updateKStar(Kstar, K, Lambda, [], [], ElbowRight, ElbowLeft);
#    LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, error  
    return LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, mae;
#, LastEventPoint, LastEvent;  
# Updates(adds columns/rows) to a Gram-submatrix Kstar
# 
def updateKStar(Kstar,K_elbowsRL,Krow,Kcol,y_elbows_transformed):
    if np.asarray(y_elbows_transformed).size == 1:
        newrow = [y_elbows_transformed];
        newrow += list(Krow)
        newcol = [1]
        newcol += list(Kcol)
        newcol.append(K_elbowsRL)
        Kstar2 = np.ndarray(shape=[Kstar.shape[0]+1,Kstar.shape[1]+1])
        Kstar2[0:-1,0:-1] = Kstar
        Kstar2[-1,:-1] = newrow
        Kstar2[:,-1] = newcol
        return Kstar2
    else:
        if Krow is not None:
            print('Krow is not none! not handled yet.');
            exit();
        else:
            #this looks like a y0 = (0, y_e^l) vector
            y0 = np.ndarray(shape=[len(y_elbows_transformed)+1])
            y0[0] = Kstar;
            y0[1:] = y_elbows_transformed;
            #this looks like the K-subset matrix, with a (1) row vector added on top
            y1 = np.ones(shape=[K_elbowsRL.shape[1]]);
            result = np.ndarray(shape=[K_elbowsRL.shape[0]+1, K_elbowsRL.shape[1]+1]);
            result[:,0] = y0;
            result[0,1:] = y1;
            result[1:,1:] = K_elbowsRL;
            return result;
            
def SolveKstar(Kstar):
    onestar = np.ones(Kstar.shape[1]);
    onestar[0] = 0;
    return np.linalg.solve(Kstar, onestar)    
    
def empty_elbows(K, y, eps, beta0, Lambda, alphas, gammas, LeftRegion, RightRegion, Center, ElbowRight, ElbowLeft, elbowPointIdx, elbowSide, eps_of_eps):
    # This seems like the support vectors * coefficients, that are used to make a prediction for a new value.
#    print('\nempty.elbows')
    betaStar = np.dot(K, alphas - gammas);
    LeftRegion = list(LeftRegion.flatten());
    RightRegion = list(RightRegion.flatten());
    # Careful here, as we want two independent instances of lists; since I use "del Center[x]", the backups need to be strongly decoupled from the working copies.
    Center = list(Center.flatten());
    ElbowLeft = list(ElbowLeft.flatten());
    ElbowRight = list(ElbowRight.flatten());
    BackupCenter = Center.copy()
    BackupRight = RightRegion.copy()
    BackupLeft = LeftRegion.copy()
    
    if len(ElbowLeft) + len(ElbowRight) != 0 :
        print('Condition not covered, as I understand, we should always have exactly 0 points on an elbow!');
        print(len(ElbowLeft) + len(ElbowRight))
        exit();
    emptyElbowsIteration=1;
    while len(ElbowLeft) + len(ElbowRight) < 2: # I don't get this condition. In the paper, they concentrate on having only unique points on the elbows, and this is not covered.
#        print('\nempty elbows iteration: ',emptyElbowsIteration);
        # Move the event point from its original list to an elbow
        if elbowPointIdx in RightRegion:
            RightRegion.remove(elbowPointIdx);
#            RightRegion = np.asarray(list(RightRegion).remove(elbowPointIdx));
        elif elbowPointIdx in LeftRegion:
            LeftRegion.remove(elbowPointIdx);
        elif elbowPointIdx in Center:
            Center.remove(elbowPointIdx);
        if elbowSide == 'ElbowRight':
            ElbowRight = [elbowPointIdx];
            # What is this const? I don't know.
            const = -1
        elif elbowSide == 'ElbowLeft':
            ElbowLeft = [elbowPointIdx];
            const = 1
        #(beta.star[c(Righti, Centeri)]-beta.star[i])/(y[c(Righti, Centeri)] - y[i] - const*eps - eps)
        eventsFromRight = [];
        cAndR = RightRegion + Center
        
        # I don't fully understand it, especially the const*eps term.
        eventsFromRight = (betaStar[cAndR] - betaStar[elbowPointIdx]) / (y[cAndR] - y[elbowPointIdx] - const*eps - eps)
        #(beta.star[c(Lefti, Centeri)]-beta.star[i])/(y[c(Lefti, Centeri)] - y[i] - const*eps + eps)
        cAndL = LeftRegion + Center
        eventsFromLeft = (betaStar[cAndL] - betaStar[elbowPointIdx]) / (y[cAndL] - y[elbowPointIdx] - const*eps + eps);
        # Here, we choose the highest lambda (that is implicitly less than the current lambda) - moving down to it forces a singular event.
        possibleLambdasForEvent = np.asarray(list(eventsFromRight)+list(eventsFromLeft));
        # TODO: note that here, I'm not sure we want eps_of_eps. This is actually some tolerance for lambda.
        possibleLambdasForEventFiltered = possibleLambdasForEvent[possibleLambdasForEvent < Lambda - eps_of_eps]
        lambdaNew = np.max(possibleLambdasForEventFiltered)
       
        beta0New = y[elbowPointIdx] - betaStar[elbowPointIdx]/lambdaNew + const*eps;
        rightCenterLeftCenter = RightRegion + Center + LeftRegion + Center;
#        print(RightRegion, Center, LeftRegion, Center)
#        print(rightCenterLeftCenter);
        # Check if we have more than one event for that new lambda.
        eventsForNewLambda = []
        for i in range(0,len(possibleLambdasForEvent)):
            if np.abs(lambdaNew - possibleLambdasForEvent[i]) < eps_of_eps:
                eventsForNewLambda.append(rightCenterLeftCenter[i]);
                
        if len(eventsForNewLambda) > 1:
            print('len(eventsForNewLambda) > 1!');
            exit();
            
        eventForNewLambda = eventsForNewLambda[0];
#        print('event for new lambda',eventForNewLambda)
            
        # check which elbow the new point will go to, and assign it
        if y[eventForNewLambda] - betaStar[eventForNewLambda] / lambdaNew - beta0New < 0:
            ElbowLeft.append(eventForNewLambda);
            newElbowSide = 'ElbowLeft';
            if eventForNewLambda in Center:
                Center.remove(eventForNewLambda);
            elif eventForNewLambda in LeftRegion:
                LeftRegion.remove(eventForNewLambda);
            else:
                RightRegion.remove(eventForNewLambda);
        else:
            ElbowRight.append(eventForNewLambda);
            newElbowSide = 'ElbowRight';
            if eventForNewLambda in Center:
                Center.remove(eventForNewLambda);
            elif eventForNewLambda in RightRegion:
                RightRegion.remove(eventForNewLambda);
            else:
                LeftRegion.remove(eventForNewLambda);
                
        # I believe this has something to do with recomputing the Thetas for the points on the elbows (since those are between 0 and 1)
        y_elbows_transformed = list(1/(y[ElbowRight] - eps)) + list(1/(y[ElbowLeft] + eps));
        kstar_param1 = np.zeros(shape=[1,1]);
        
        # This is a quite hard port from R to python.
        # Basically, we want to extract a subset of the Gram-matrix consisting ONLY of the elbow points. So the diagonal is 1's, before multiplying by y_elbows_transformed
        pts_of_interests = ElbowRight+ElbowLeft;
        K_subset_of_interest = np.ndarray(shape=[len(pts_of_interests),len(pts_of_interests)])
        for rowidx in range(0,K_subset_of_interest.shape[0]):
            K_subset_of_interest[rowidx] = K[pts_of_interests][rowidx,pts_of_interests];
        kstar_param2 = K_subset_of_interest
        for rowidx in range(0,kstar_param2.shape[0]):
            kstar_param2[rowidx] *= y_elbows_transformed[rowidx]
        Kstar = updateKStar(kstar_param1, kstar_param2, None, None, y_elbows_transformed)
        bm = SolveKstar(Kstar)[1:];

        # We know that all thetas added up to 0 before we moved the two new shoulder pts.
        # So, check their new thetas
        theta1 = alphas[elbowPointIdx] - gammas[elbowPointIdx];
        theta2 = alphas[eventForNewLambda] - gammas[eventForNewLambda];
        # Why the abs of the thetas though? I don't get it.
        
        absDiffThetas = abs(abs(theta1)-abs(theta2));
        # I have no idea about these conditions. They do not seem to be mentioned in the paper, either.
        # They may be needed after the initial iteration..
        drop = 0;
        if len(ElbowLeft) == len(ElbowRight):
            if absDiffThetas > eps_of_eps:
#                print('drop1');
                drop = 1;
        if len(ElbowLeft) != len(ElbowRight):
            if absDiffThetas < eps_of_eps:
#                print('drop2');
                drop = 1;
        strangemetric = np.asarray(list(1-2*alphas[ElbowRight])+list(2*gammas[ElbowLeft]-1))/bm
        if drop == 0:
            if np.max(strangemetric) > 0:
#                print('drop3',strangemetric);
                drop = 1;
        
        if drop:
#            print('Drop the elbows.');
            ElbowLeft = []
            ElbowRight = []
            #careful here, we want a hard copy
            Center = BackupCenter.copy();
            RightRegion = BackupRight.copy();
            LeftRegion = BackupLeft.copy();
            
        # This is easier now: just update the l+1'th event point to be the "starting elbow point" for the next iteration of path, and update the other params.
        elbowPointIdx = eventForNewLambda;
        Lambda = lambdaNew;
        beta0 = beta0New;
        elbowSide = newElbowSide;
        emptyElbowsIteration+=1;
        
#    print('Empty elbows successful!')
    return { 'right' : RightRegion, 'left' : LeftRegion, 'elbowleft' : ElbowLeft, 'elbowright' : ElbowRight, 'center' : Center, 
             'lambda' : Lambda, 'alphas' : alphas, 'gammas' : gammas, 'betal0' : beta0 * Lambda }

def init_event(K, y, eps, beta0, alphas, gammas, LeftRegion, RightRegion, ElbowLeft, ElbowRight, Center, eps_of_eps):
    # find the range in which beta0 can range (the dc signal), without affecting the value of the cost function.
    # this is either when the eps boundary hits a lower label, or a center point will hit the elbow
    leftAndElbow = np.concatenate([ElbowLeft,LeftRegion]);
    beta0min = np.max(np.asarray(y)[leftAndElbow])+eps
    
    if len(Center) > 0:
        beta0min = np.max([beta0min, np.max(y[Center])-eps]);
    rightAndElbow = np.concatenate([ElbowRight,RightRegion]);
    beta0max = np.min(y[rightAndElbow])-eps;
    if len(Center) > 0:
        beta0max = np.min([beta0max, np.min(y[Center])+eps])
    
    # fix the value of beta0 to the closer value between the betamin and betamax. This will definitely put a value on a shoulder.
    if np.abs(beta0min-beta0) > np.abs(beta0max-beta0):
        beta0fixed = beta0max;
    else:
        beta0fixed = beta0min;
    # implement elbowPointIdx
    # implement elbowSide
    absErrors = np.abs(np.abs(y - beta0fixed)-eps);
    elbowPointIdx = np.argmin(absErrors);
    if y[elbowPointIdx] > beta0fixed:
        elbowSide = 'ElbowRight'
    else:
        elbowSide = 'ElbowLeft'
    # TODO: what does this do?
    res = empty_elbows(K,y,eps,beta0fixed,1e50,alphas,gammas,LeftRegion,RightRegion,Center,ElbowRight,ElbowLeft, elbowPointIdx, elbowSide, eps_of_eps)
    res.update({'alphas' : alphas, 'gammas' : gammas})
    return res
    
def calcStats(y, y_modeled, eps):
    errors = y_modeled - y
    right = errors - eps
    left = errors + eps
    for i in range(0,len(errors)):
        if abs(left[i]) < abs(right[i]):
            errors[i] = left[i]
        else:
            errors[i] = right[i]
    mean_average_error = np.sum(np.abs(errors)) / len(errors)
#    print('mae:',mean_average_error)
#    print('errors:',errors)
    return mean_average_error, errors
    
# Removes a no-longer-elbow point from the Kstar matrix.
def DowndateKstar(Kstar, index):
    Kstar = np.delete(Kstar, (index), axis=0)
    Kstar = np.delete(Kstar, (index), axis=1)
    return Kstar

def svrpath(x,y,K,eps,maxIterations, lambdamin, rho, LambdaLast):
    Lambda = LambdaLast
    print('Brand New Comment')
    start = time.time()
    n = K.shape[0];
    # slack coeffs corresponding to the elbows or to the outside regions (0,1 for elbows) (1 for outside regions) (0 for inside)
    AllAlphas = []
    AllGammas = []
    AllBetaZeros = []
    maes = []
    errTerms = []
    lambdas = []
    elbowL = []
    elbowR = []
    centers = []
    rights = []
    lefts = []
    y = np.asarray(y)
    
    # (Was hardcoded in the PATH algorithm!) Tolerance of determining whether a point is at the elbow.
    # Currently, we take either 0.01*eps, 10e-10(original algo), or the smallest difference between two labels.
    #epsilon_of_epsilon = np.min([eps*1e-2, 10e-10, np.min(np.ediff1d(sorted_vals))]);
    epsilon_of_epsilon = eps*1e-3
    
    # The initial event.
    moveIdx = 0 
#    LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, error = svrinit(K,y,eps, epsilon_of_epsilon);
    LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, error = svrload(K,y,eps);
    betal0 = beta0
    # Lambda will be infinite; this means that f(x) -> beta_0 (constant term)
#    print('beta0',beta0)
#    if len(ElbowLeft) + len(ElbowRight) < 2:
#        initial_state = init_event(K, y, eps, beta0, alphas, gammas, LeftRegion, RightRegion, ElbowLeft, ElbowRight, Center, epsilon_of_epsilon)
#    else:
#        print('init_event quadratic, not implemented.')
        
#    ElbowLeft = initial_state['elbowleft']
#    ElbowRight = initial_state['elbowright']
#    Center = initial_state['center']
#    RightRegion = initial_state['right']
#    LeftRegion = initial_state['left']
#    betal0 = initial_state['betal0']
#    Lambda = initial_state['lambda']
    
    # I believe this has something to do with recomputing the Thetas for the points on the elbows (since those are between 0 and 1)
    y_elbows_transformed = list(1/(y[ElbowRight] - eps)) + list(1/(y[ElbowLeft] + eps));
    kstar_param1 = np.zeros(shape=[1,1]);
    print('Brand New Comment #2')    
    # This is a quite hard snippet to port from R to python.
    # Basically, we want to extract a subset of the Gram-matrix consisting ONLY of the elbow points. So the diagonal is 1's, before multiplying by y_elbows_transformed
    # It also looks like the interpolating function's value, for given support vectors
    pts_of_interests = ElbowRight+ElbowLeft;
    K_subset_of_interest = np.ndarray(shape=[len(pts_of_interests),len(pts_of_interests)])
    for rowidx in range(0,K_subset_of_interest.shape[0]):
        K_subset_of_interest[rowidx] = K[pts_of_interests][rowidx,pts_of_interests];
    kstar_param2 = K_subset_of_interest
    for rowidx in range(0,kstar_param2.shape[0]):
        kstar_param2[rowidx] *= y_elbows_transformed[rowidx]
    Kstar = updateKStar(kstar_param1, kstar_param2, None, None, y_elbows_transformed)
#    bm = SolveKstar(Kstar)[1:];
    
    elbows_ordered = list(ElbowRight) + list(ElbowLeft)
    
    fl = (np.dot(K, (alphas-gammas)) + betal0) / Lambda
    mae, errors = calcStats(y, fl, eps)
    print('Brand New Comment #3')    
    # The main body starts here.
    while moveIdx < maxIterations and Lambda > lambdamin:
        moveIdx += 1
        print('\n\n================= Lambda Iteration',moveIdx,'===================')
            
        if len(ElbowLeft) > 0 or len(ElbowRight) > 0:
            print('Nonempty elbows, do standard algo')
            bstar = SolveKstar(Kstar)
            # very closely related to the update rule of beta0 for the next iteration
            b0 = bstar[0]
            # very closely related to the update rule of thetas (for the elbows) for the next iteration
            b = bstar[1:]
            # very closely related to the update rule of f itself for the next iteration
#            print('b0',b0)
#            print('b',b)
            gl = np.dot( K[:,elbows_ordered], b) + b0
            dl = fl - gl
            # the math says that we can infinitely update with no real effect if fl = gl. ( eq 2.14)
            isImmobile = (np.sum(abs(dl))/n < rho)
            
            # update lambdas for points on the shoulders that are about to exit and enter a new region.
            # Note that elbows_ordered may not be actually ordered at this point, so we need to extract the points correctly.
            rightElbowIdces = np.nonzero(np.in1d(elbows_ordered,ElbowRight))
            leftElbowIdces = np.nonzero(np.in1d(elbows_ordered,ElbowLeft))
            vecR = -alphas[ElbowRight] / b[rightElbowIdces]
            vecL = gammas[ElbowLeft] / b[leftElbowIdces]   # TODO: /mnt/hdd/work-projects/svrpath-python/svrpath.py:379: RuntimeWarning: divide by zero encountered in true_divide
            lambdasRight = np.concatenate((vecR,vecL)) + Lambda
            lambdasRight[np.abs(b) < rho] = 1
            lambdasLeft = np.asarray(list(1/b[rightElbowIdces]) + list(-1/b[leftElbowIdces])) + lambdasRight
            lambdasLeft[np.abs(b) < rho] = 1
            lambdasForChecking = np.asarray(list(lambdasRight) + list(lambdasLeft))
            lambdasForPointsExitingShoulders = np.max(np.concatenate(([-1], lambdasForChecking[lambdasForChecking < Lambda - rho])))
            
#            print('lambdas for exiting',lambdasForChecking.shape)
#            print(np.max(lambdasForChecking[lambdasForChecking < Lambda]))
            print('lambda',Lambda)
            #print('new lambda',lambdaNew)
            
            if isImmobile and lambdasForPointsExitingShoulders < rho:
                print('Immobile system! Breaking because further updates will have no effect.')
                break;
            if not isImmobile:
                # Update lambdas for points that are about to enter shoulders. Pg 1642, numberless formula (2nd and 3rd ones on the page)
                numerator = Lambda * dl[list(RightRegion) + list(LeftRegion)]
                denominator = y[list(RightRegion) + list(LeftRegion)];
                # Fix the costless margin for the Right and Left region points
                denominator[0:len(RightRegion)] -= eps;
                denominator[len(RightRegion):] += eps;
                denominator -= gl[list(RightRegion) + list(LeftRegion)]
                lambdasForPointsEnteringShoulders = numerator  / denominator
                
                # Center (viewed from right and left perspective)
                num2 = Lambda * dl[list(Center)+list(Center)]
                den2 = y[list(Center)+list(Center)];
                den2[0:len(Center)] -= eps;
                den2[len(Center):] += eps
                den2 -= gl[list(Center)+list(Center)]
                lambdasForPointsEnteringShoulders = np.concatenate((lambdasForPointsEnteringShoulders, num2/den2))
                RiLeCC = list(RightRegion)+list(LeftRegion)+list(Center)+list(Center)
                lambdasForPointsEnteringShoulders[np.abs(  np.abs(y[RiLeCC] - gl[RiLeCC]) - eps) < rho] = float('-inf')
            else: # isImmobile
                lambdasForPointsEnteringShoulders = -1;
                    
            # The maximum lambda then signifies the first event that will happen - so mark it as the next lambda.
            lambdaNew = lambdamin
            if lambdasForPointsEnteringShoulders.size > 0:
#                print('lambdas for entry',lambdasForPointsEnteringShoulders)
                filteredForEntering = lambdasForPointsEnteringShoulders[lambdasForPointsEnteringShoulders < Lambda - rho];
                if len(filteredForEntering) > 0:
                    lambdaNew = np.max([lambdaNew, np.max(lambdasForPointsEnteringShoulders[lambdasForPointsEnteringShoulders < Lambda - rho])])                        
                    lambdaEntering = np.max(filteredForEntering)
                else:
                    lambdaEntering = float('-inf')
#                print('lambdas for entry filtered',filteredForEntering)
            else:
                lambdaEntering = float('-inf')
            if lambdasForPointsExitingShoulders.size > 0:
                filteredForExiting = lambdasForPointsExitingShoulders[lambdasForPointsExitingShoulders < Lambda - rho]
                if len(filteredForExiting) > 0:
                    lambdaNew = np.max([lambdaNew, np.max(filteredForExiting)])
                    lambdaExiting = np.max(lambdasForPointsExitingShoulders[lambdasForPointsExitingShoulders < Lambda - rho])
                else:
                    lambdaExiting = float('-inf')
            else:
                lambdaExiting = float('-inf')
#            print('set new lambda',lambdaNew)
#            print('lambdaentry',lambdaEntering)
#            print('lambdaexit',lambdaExiting)
            # Use update rule for the svm parameters, and for beta0.
            rightElbowIdces = np.nonzero(np.in1d(elbows_ordered,ElbowRight))
            leftElbowIdces = np.nonzero(np.in1d(elbows_ordered,ElbowLeft))
            if len(ElbowRight) > 0:
                alphas[ElbowRight] -= (Lambda - lambdaNew) * b[rightElbowIdces]
            if len(ElbowLeft) > 0:
                gammas[ElbowLeft] += (Lambda - lambdaNew) * b[leftElbowIdces]
            betal0 -= (Lambda - lambdaNew) * b0
            # Update the interpolation function (formula 2.14)
            fl = Lambda/lambdaNew * dl + gl
            if lambdaEntering >= lambdaExiting - rho:
#                print('event: entering a shoulder')
                eventPointIdx = np.argwhere(lambdaEntering == lambdasForPointsEnteringShoulders)
#                print('right',RightRegion,'left',LeftRegion,'Center',Center)
#                print('eventPointIdx',eventPointIdx)
                observedEvent = RiLeCC[int(eventPointIdx)]
#                print('observedEvent',observedEvent)                
                if observedEvent in LeftRegion:
#                    print('observed movement left->elbowleft',observedEvent)
                    krow = K[observedEvent, elbows_ordered] / (y[observedEvent] + eps)
                    kcol = K[elbows_ordered, observedEvent] / (y[elbows_ordered] + eps)
                    for item in ElbowRight:
                        idx = np.argwhere(elbows_ordered == item) # we should get exactly one match here; no item shows up twice in ordered elbows
                        kcol[idx[0]] = K[item, observedEvent] / (y[item] - eps)
                    # Update parameters.
                    kstar_param2 = K[observedEvent,observedEvent] / (y[observedEvent]+eps)
                    Kstar = updateKStar(Kstar, kstar_param2, krow, kcol, 1/(y[observedEvent]+eps))
                    idxToDel = np.argwhere(LeftRegion == observedEvent)
                    del LeftRegion[int(idxToDel)]
                    elbows_ordered.append(observedEvent)
                    ElbowLeft.append(observedEvent)
                    moveto = 'ElbowLeft'
                    movefrom = 'LeftRegion'   
                  
                if observedEvent in RightRegion:
#                    print('observed movement right->elbowright',observedEvent)
                    krow = K[observedEvent, elbows_ordered] / (y[observedEvent] - eps)
                    kcol = K[elbows_ordered, observedEvent] / (y[elbows_ordered] - eps)
                    for item in ElbowLeft:
                        idx = np.argwhere(elbows_ordered == item) # we should get exactly one match here; no item shows up twice in ordered elbows
                        kcol[idx[0]] = K[item, observedEvent] / (y[item] + eps)
                    # Update parameters.
                    kstar_param2 = K[observedEvent,observedEvent] / (y[observedEvent]-eps)
                    Kstar = updateKStar(Kstar, kstar_param2, krow, kcol, 1/(y[observedEvent]-eps))
                    idxToDel = np.argwhere(RightRegion == observedEvent)
                    del RightRegion[int(idxToDel)]
                    elbows_ordered.append(observedEvent)
                    ElbowRight.append(observedEvent)
                    moveto = 'ElbowRight'
                    movefrom = 'RightRegion'
                if observedEvent in Center:
#                    print('for the if',eventPointIdx, len(RightRegion) + len(LeftRegion) + len(Center))
                    if eventPointIdx >= len(RightRegion) + len(LeftRegion) + len(Center):
#                        print('observed movement center->elbowleft',observedEvent)
                        krow = K[observedEvent, elbows_ordered] / (y[observedEvent] + eps)
                        kcol = K[elbows_ordered, observedEvent] / (y[elbows_ordered] + eps)
                        for item in ElbowRight:
                            idx = np.argwhere(elbows_ordered == item) # we should get exactly one match here; no item shows up twice in ordered elbows
                            kcol[idx[0]] = K[item, observedEvent] / (y[item] - eps)
                        # Update parameters.
                        kstar_param2 = K[observedEvent,observedEvent] / (y[observedEvent]+eps)
                        Kstar = updateKStar(Kstar, kstar_param2, krow, kcol, 1/(y[observedEvent]+eps))
                        elbows_ordered.append(observedEvent)
                        ElbowLeft.append(observedEvent)
                        moveto = 'ElbowLeft'  
                    else:
#                        print('observed movement center->elbowright',observedEvent)
                        krow = K[observedEvent, elbows_ordered] / (y[observedEvent] - eps)
                        kcol = K[elbows_ordered, observedEvent] / (y[elbows_ordered] - eps)
                        for item in ElbowLeft:
                            idx = np.argwhere(elbows_ordered == item) # we should get exactly one match here; no item shows up twice in ordered elbows
                            kcol[idx[0]] = K[item, observedEvent] / (y[item] + eps)
                        # Update parameters.
                        kstar_param2 = K[observedEvent,observedEvent] / (y[observedEvent]-eps)
                        Kstar = updateKStar(Kstar, kstar_param2, krow, kcol, 1/(y[observedEvent]-eps))
                        elbows_ordered.append(observedEvent)
                        ElbowRight.append(observedEvent)
                        moveto = 'ElbowRight'
                    idxToDel = np.argwhere(Center == observedEvent)
                    del Center[int(idxToDel)]
                    movefrom='Center'
                Lambda = lambdaNew
            else: # not np.max(lambdasForPointsEnteringShoulders) >= np.max(lambdasForPointsExitingShoulders) - rho:
#                print('event point - exiting a shoulder')
                Leaveout = []
                idrop = []
                # wrong comment? Detect additional events for points entering shoulders (remember, rho controls tolerance of simultaneous events)
                if np.min(np.abs(lambdaExiting - lambdasRight)) < rho:
                    eventPointIdx = np.asarray(ElbowRight + ElbowLeft)[np.abs(lambdaExiting - lambdasRight) < rho];
                    Leaveout += [False] * len(eventPointIdx)
                    idrop += list(eventPointIdx)
#                    print('drop right',eventPointIdx)
                if np.min(np.abs(lambdaExiting - lambdasLeft)) < rho:
                    eventPointIdx = np.asarray(ElbowRight + ElbowLeft)[np.abs(lambdaExiting - lambdasLeft) < rho];
                    Leaveout += [True] * len(eventPointIdx)
                    idrop += list(eventPointIdx)
#                    print('drop left',eventPointIdx)
                observedEvents = idrop
                movefrom = []
                moveto = []
                for j in range(0,len(idrop)):
                    if Leaveout[j]:
                        if observedEvents[j] in ElbowRight:
#                            print('Moving a point from ElbowLeft to LeftRegion')
                            movefrom.append('ElbowRight')
                            moveto.append('RightRegion')
                            RightRegion.append(observedEvents[j])
                            idx = np.argwhere(np.asarray(ElbowRight) == observedEvents[j])
                            del ElbowRight[int(idx)]
                            # TODO: what is mi?
                            mi = np.argwhere(Kstar[:,0] == 1/(y[observedEvents[j]] - eps))
                        else: # ! observedEvents[j] in ElbowRight:
#                            print('Moving a point from ElbowLeft to LeftRegion')
                            movefrom.append('ElbowLeft')
                            moveto.append('LeftRegion')
                            LeftRegion.append(observedEvents[j])
                            idx = np.argwhere(np.asarray(ElbowLeft) == observedEvents[j])
                            del ElbowLeft[int(idx)]
                            mi = np.argwhere(Kstar[:,0] == 1/(y[observedEvents[j]] + eps))
                    else:
                        moveto.append('Center')
                        Center.append(observedEvents[j])
                        
                        if observedEvents[j] in ElbowRight:
#                            print('Moving a point from ElbowRight to Center')
                            movefrom.append('ElbowRight')
                            idx = np.argwhere(np.asarray(ElbowRight) == observedEvents[j])
                            del ElbowRight[int(idx)]
                            mi = np.argwhere(Kstar[:,0] == 1/(y[observedEvents[j]] - eps))
                        else:
#                            print('Moving a point from ElbowLeft to Center')
                            movefrom.append('ElbowLeft')
                            idx = np.argwhere(np.asarray(ElbowLeft) == observedEvents[j])
                            del ElbowLeft[int(idx)]
                            mi = np.argwhere(Kstar[:,0] == 1/(y[observedEvents[j]] + eps))
                            
                    # Remove the corresponding elbow points from the Kstar matrix and from the elbows_ordered
#                    print('downdating kstar')
                    Kstar = DowndateKstar(Kstar, mi)
                    for ptToRemove in observedEvents:
                        idx = np.argwhere(ptToRemove == elbows_ordered)
                        if idx.size > 0:
                            del elbows_ordered[int(idx)]
                Lambda = lambdaNew
                            
        else: # when ElbowLeft and ElbowRight are empty
#            print('Empty elbows, do the empty elbows algo')
            if np.sum(gammas) - np.sum(alphas) > 0.1:
                print('Unbalanced data in interior empty elbows situation.')
                return;
            print('movefrom',movefrom,type(movefrom))
            
            # We previously have moved elbows, so call the routine that handles the empty elbow situation.
            ee = empty_elbows(K, y, eps, betal0/Lambda, Lambda, alphas, gammas, 
                              np.asarray(LeftRegion), np.asarray(RightRegion), np.asarray(Center), np.asarray([]), np.asarray([]), 
                              observedEvents[1], movefrom[1], epsilon_of_epsilon);
            ee2 = empty_elbows(K, y, eps, betal0/Lambda, Lambda, alphas, gammas, 
                              np.asarray(LeftRegion), np.asarray(RightRegion), np.asarray(Center), np.asarray([]), np.asarray([]), 
                              observedEvents[0], movefrom[0], epsilon_of_epsilon);
            if abs(ee['lambda'] - ee2['lambda'] > 1e-10):
                print("EE1 lambda != EE2 lambda\n")
                print("empty elbows from left or right gives different lambdas... but what does it mean?")
                raise Exception("EE1 lambda != EE2 lambda");
            
            print('new elbows.')                
            ElbowLeft = ee['elbowleft']
            ElbowRight = ee['elbowright']
            Center = ee['center']
            RightRegion = ee['right']
            LeftRegion = ee['left']
            Lambda = ee['lambda']
            betal0 = ee['betal0']
            alphas = ee['alphas']
            gammas = ee['gammas']
                
            # Update parameters.
            rightEvents = np.asarray(list(y[ElbowRight])) - eps
            leftEvents = np.asarray(list(y[ElbowLeft])) + eps
            
            y_elbows_transformed = 1 / np.asarray(list(rightEvents) + list(leftEvents))
            pts_of_interests = ElbowRight+ElbowLeft;
            K_subset_of_interest = np.ndarray(shape=[len(pts_of_interests),len(pts_of_interests)])
            for rowidx in range(0,K_subset_of_interest.shape[0]):
                K_subset_of_interest[rowidx] = K[pts_of_interests][rowidx,pts_of_interests];
            kstar_param2 = K_subset_of_interest
            for rowidx in range(0,kstar_param2.shape[0]):
                kstar_param2[rowidx] *= y_elbows_transformed[rowidx]
            Kstar = updateKStar(np.zeros(shape=[1,1]), kstar_param2, None, None, y_elbows_transformed)
            elbows_ordered = list(ElbowRight)+list(ElbowLeft)
            fl = (np.dot(K, (alphas-gammas)) + betal0) / Lambda
            movefrom = ' ' * (len(ElbowRight) + len(ElbowLeft))
            moveto = 'ElbowRight' * len(ElbowRight) + 'ElbowLeft' * len(ElbowLeft)
            observedEvents = ElbowRight + ElbowLeft
                
        # End of iteration
        mae, errors = calcStats(y, fl, eps)
        
        AllAlphas.append(alphas)
        AllGammas.append(gammas)
        AllBetaZeros.append(beta0)
        maes.append(mae)
        lambdas.append(Lambda)
        errTerms.append(errors)
        elbowL.append(ElbowLeft)
        elbowR.append(ElbowRight)
        centers.append(Center)
        rights.append(RightRegion)
        lefts.append(LeftRegion)

    print('Algo done.')
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    timeTaken = ("%d:%02d:%02d" % (h, m, s))
    return { 'alphas' : np.asarray(AllAlphas), 'gammas' : np.asarray(AllGammas), 'beta0' : np.asarray(AllBetaZeros), 'maes' : np.asarray(maes), 'errors' : np.asarray(errTerms), 
            'lambdas' : lambdas, 'ElbowLeft' : elbowL, 'ElbowRight' : elbowR, 'Center' : centers, 'RightRegion' : rights, 'LeftRegion' : lefts, 'TimeTaken' : timeTaken }
