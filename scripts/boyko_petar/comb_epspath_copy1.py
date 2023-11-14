#TODO: what about 2d eps path
import numpy as np
#import scipy.linalg
import sklearn.svm
import time

import matplotlib.pyplot as plt

#TODO drop import  of svrpath, add to one structure
import lambda_path
#import test_svrpath

on_margin_tolerance = 1e-15
globaltype = np.float64
#TODO: parametrize rho.
rho = 1e-14
DebugPrint = False;
PlotFiguresPerIteration = False

def update_eps(eps, y, fl, K, Kstar, Lambda, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, iteration, LastEventPoint, LastEvent):  
    if DebugPrint:
        print('\nupdate_eps')
    backup = { 'eps' :eps, 'y' :y, 'fl':fl, 'K':K, 'Kstar':Kstar, 'Lambda':Lambda, 'LeftRegion':LeftRegion, 'ElbowLeft':ElbowLeft, 'Center':Center, 'ElbowRight':ElbowRight, 
              'RightRegion':RightRegion, 'alphas':alphas.copy(), 'gammas':gammas.copy(), 'beta0':beta0, 'iteration':iteration, 'LastEventPoint':LastEventPoint, 'LastEvent':LastEvent }
    
    
    b = find_b(eps=eps, Kstar=Kstar, ElbowLeft=ElbowLeft, ElbowRight = ElbowRight)
    eps_ER_to_R, eps_ER_to_C, eps_EL_to_L, eps_EL_to_C = exit_elbows_events(eps=eps, alphas = alphas, gammas=gammas, Lambda=Lambda, b = b, ElbowLeft=ElbowLeft, ElbowRight=ElbowRight)
    hl = find_h(K, Kstar, b, Lambda, ElbowLeft, ElbowRight)
    if DebugPrint:
        print('fl:',fl)
    eps_R_to_ER, eps_C_to_ER, eps_L_to_EL, eps_C_to_EL = enter_elbows_events(fl=fl, eps=eps, y=y, hl=hl, LeftRegion=LeftRegion, ElbowLeft=ElbowLeft, Center=Center, ElbowRight=ElbowRight, RightRegion=RightRegion)
    epsEnter = np.concatenate((eps_R_to_ER, eps_C_to_ER, eps_L_to_EL, eps_C_to_EL))
    epsExit = np.concatenate((eps_ER_to_R, eps_ER_to_C, eps_EL_to_L, eps_EL_to_C))
    
    validEventsEnter = epsEnter[epsEnter < eps - rho]
    eps_enter = np.max(validEventsEnter)
    validEventsExit = epsExit[epsExit < eps - rho]
    eps_exit = float('-Inf')
    if len(validEventsExit) > 0:
        eps_exit = np.max(validEventsExit)
    #TODO: possibly choose eps_enter with priority if eps_enter > eps_exit - rho (from Lambda path)
    eps_new = np.max([eps_enter, eps_exit])
    if DebugPrint:
        print('\neps_l:',eps,'eps_enter:',eps_enter,',eps_exit:',eps_exit,'eps_new:',eps_new,'total valid events enter exit:',validEventsEnter.shape,validEventsExit.shape)

#    eps_new = eps_new*0.5 + eps*0.5
#    eps_new = eps - 1
    
    if DebugPrint:
        print('before update alphas:',list(alphas),'\ngammas:',list(gammas))
    alphas, gammas, beta0 = updateAlphasGammasBeta0(b, eps, eps_new, beta0, alphas, gammas, ElbowRight, ElbowLeft)   
    if DebugPrint:
        print('max alpha %e point index: %d' % ((np.max(alphas)),np.argmax(alphas)))
        print('min alpha %e point index: %d' % ((np.min(alphas)),np.argmin(alphas)))
        print('max gamma %e point index: %d' % ((np.max(gammas)),np.argmax(gammas)))
        print('min gamma %e point index: %d' % ((np.min(gammas)),np.argmin(gammas)))
    if DebugPrint:
        print('before register events alphas:',list(alphas),'\ngammas:',list(gammas))
        print('sum of alphas gammas %e ' % np.sum(alphas - gammas))
        print('Last iteration\'s event',LastEventPoint, LastEvent)
    
    # TODO: EXPERIMENTAL: if the KKT conditions are unfulfilled, it is possible that the last registered event needs to be reversed.
    if (alphas <= 0 - rho).any() or (gammas <= 0 - rho).any() or (alphas >= 1 + rho).any() or (gammas >= 1 + rho).any():
        print('\n\nEXPERIMENTAL ALGORITHM')
        if alphas[LastEventPoint] <= 0 - rho or gammas[LastEventPoint] <= 0 - rho or alphas[LastEventPoint] >= 1 + rho or gammas[LastEventPoint] >= 1 + rho:
            eps = backup['eps']
            fl = backup['fl']
            Kstar = backup['Kstar']
            Lambda = backup['Lambda']
            alphas = backup['alphas']
            gammas = backup['gammas']
            beta0 = backup['beta0']
            # remove the last event point and place it back correspondingly
            if LastEvent[0] == 'C':
                Center.append(LastEventPoint)
            elif LastEvent[0] == 'L':
                LeftRegion.append(LastEventPoint)
            elif LastEvent[0] == 'R':
                RightRegion.append(LastEventPoint)
            elif LastEvent[0] == 'ER':
                ElbowRight.append(LastEventPoint)
            elif LastEvent[0] == 'EL':
                ElbowLeft.append(LastEventPoint)
            else:
                print(LastEvent[0])
                assert(False)
            if LastEvent[1] == 'ER':
                newElbowRight = ElbowRight.copy()
                print('ER',newElbowRight)
                print('argwhere',np.argwhere(newElbowRight == LastEventPoint)[0])
                del newElbowRight[np.argwhere(newElbowRight == LastEventPoint)[0]]
                ElbowRight = newElbowRight
            elif LastEvent[1] == 'EL':
                newElbowLeft = ElbowLeft.copy()
                print('ER',newElbowLeft)
                print('argwhere',np.argwhere(newElbowLeft == LastEventPoint)[0])
                del newElbowLeft[np.argwhere(newElbowLeft == LastEventPoint)[0]]
                ElbowLeft = newElbowLeft
            elif LastEvent[1] == 'R':
                newRight = RightRegion.copy()
                del newRight[np.argwhere(newRight == LastEventPoint)[0]]
                RightRegion = newRight
            elif LastEvent[1] == 'L':
                newLeft = LeftRegion.copy()
                del newLeft[np.argwhere(newLeft == LastEventPoint)[0]]
                LeftRegion = newLeft
            else:
                assert(False)
            if DebugPrint:    
                print('nER',ElbowRight)
                print('nEL',ElbowLeft)
            Kstar = updateKstar(K, np.zeros(shape=[1,1], dtype=globaltype), Lambda, [], [], ElbowRight, ElbowLeft)
            if DebugPrint:  
                print('newKstar',Kstar)
            return update_eps(eps=eps, y=y, fl=fl, K=K, Kstar=Kstar, Lambda=Lambda, 
                                                                              LeftRegion=LeftRegion, ElbowLeft=ElbowLeft, Center=Center, ElbowRight=ElbowRight, RightRegion=RightRegion, 
                                                                              alphas=alphas, gammas=gammas, beta0=beta0, iteration=iteration, LastEventPoint=LastEventPoint, LastEvent=LastEvent)
        else:
            print('EXPERIMENTAL ALGORITHM #2')
#            # try to hard-set the violating point as an outside elbow, and recompute from old eps
#            if (alphas <= 0 - rho).any():
#                pointidx = np.argmin(alphas)
#                newElbowRight = ElbowRight.copy()
#                print('ER',newElbowRight)
#                print('argwhere',np.argwhere(newElbowRight == LastEventPoint)[0])
#                del newElbowRight[np.argwhere(newElbowRight == LastEventPoint)[0]]
#                ElbowRight = newElbowRight
#                Center.append(pointidx)
#                assert(False)
#            elif (alphas >= 1 + rho).any():
#                pointidx = np.argmax(alphas)
#            elif (gammas >= 1 + rho).any():
#                assert(False)
#            elif (gammas[LastEventPoint] <= 0 - rho).any():
#                assert(False)
#            else:
#                assert(False)
            
            
        
    assert((alphas >= 0 - rho*1e6).all())
    assert((alphas <= 1 + rho*1e6).all())
    assert((gammas >= 0 - rho*1e6).all())
    assert((gammas <= 1 + rho*1e6).all())
    
    
    if PlotFiguresPerIteration:
        plt.figure(iteration)
        plt.subplot(2,1,1)
        plt.plot(fl,c='blue')
        plt.plot(fl+eps,'b--',)
        plt.plot(fl-eps,'b--',)
    fl = updateFl(fl, hl, eps, eps_new)
    if PlotFiguresPerIteration:
        plt.plot(fl,c='green')
        plt.plot(fl+eps_new,'g--')
        plt.plot(fl-eps_new,'g--')
        plt.plot(beta0,'g--')
        plt.scatter(np.arange(len(y)),y,c='magenta',s=2)
        plt.ylim(np.min(y)-1)
        plt.show()
    
    
    LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, LastEventPoint, LastEvent = registerEvents(eps, eps_new, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, 
                                                                            eps_R_to_ER, eps_C_to_ER, eps_L_to_EL, eps_C_to_EL,
                                                                            eps_ER_to_R, eps_ER_to_C, eps_EL_to_L, eps_EL_to_C, alphas, gammas, validEventsEnter, validEventsExit)
    if DebugPrint:
        print('after register events alphas:',list(alphas),'\ngammas:',list(gammas))
        print('sum of alphas gammas after registerEvents: %e ' % np.sum(alphas - gammas))
    Kstar = updateKstar(K, np.zeros(shape=[1,1], dtype=globaltype), Lambda, [], [], ElbowRight, ElbowLeft)
    
    
    assert(np.sum(alphas - gammas) < 1e-10)
    
    #update the following variables:
    # +eps, fl, Kstar, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0?
    # TODO: resolve other simultaneous events that have happened? especially points on elbows that have just left it (from the same side as they entered) or points juuust outside elbows that are reaching in.
    
    return eps_new, fl, Kstar, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, LastEventPoint, LastEvent

def updateKstar(K, Kstar_old, Lambda, ElbowRight, ElbowLeft, ptsToAddER, ptsToAddEL):
    if DebugPrint:
        print('\nupdateKstar')
    newElbowRight = ElbowRight + ptsToAddER;
    newElbowLeft = ElbowLeft + ptsToAddEL;
    n = len(newElbowLeft) + len(newElbowRight) + 1
    Kstar_new = np.ndarray(shape=[n,n],dtype=globaltype)
    
#    Kstar_new[1:1+len(newElbowRight),0] = 1
#    Kstar_new[-len(newElbowRight):,0] = -1
    Kstar_new[1:,0] = 1
    Kstar_new[0,1:] = 1
    Kstar_new[0,0] = 0
    offset = len(newElbowRight)
    for kstarIdx, item in enumerate(newElbowRight):
        for kstarIdx2, item2 in enumerate(newElbowRight):
            Kstar_new[kstarIdx+1, kstarIdx2+1] = K[item,item2] / Lambda
        
        for kstarIdx2, item2 in enumerate(newElbowLeft):
            Kstar_new[kstarIdx+1, kstarIdx2+1+offset] = K[item,item2] / Lambda
            
    for kstarIdx, item in enumerate(newElbowLeft):
        for kstarIdx2, item2 in enumerate(newElbowRight):
            Kstar_new[kstarIdx+1+offset, kstarIdx2+1] = K[item,item2] / Lambda
            
        offset = len(newElbowRight)
        for kstarIdx2, item2 in enumerate(newElbowLeft):
            Kstar_new[kstarIdx+1+offset, kstarIdx2+1+offset] = K[item,item2] / Lambda
    
    if DebugPrint:
        print('Kstar',Kstar_new)
    return Kstar_new

def svrinit(K, y, Lambda):
    if DebugPrint:
        print('\nsvrinit')
    # The initial solution is trivial: epsilon is any value larger than ymax-ymin /2, beta0 is ymax - ymin / 2, and all parameters are 0, because they are all in the center region.
    Center = list(np.arange(0,len(y)));
    LeftRegion = []
    RightRegion = []
    ElbowLeft = []
    ElbowRight = []
    alphas = np.zeros(len(y), dtype=globaltype)
    gammas = np.zeros(len(y), dtype=globaltype)
    beta0 = (np.max(y) + np.min(y)) / 2
    mae = np.sum(np.abs(y - beta0));
    eps = (np.max(y) - np.min(y)) / 2
    ElbowRight = list(np.argwhere(np.abs(y - (beta0 + eps)) < on_margin_tolerance )[0])
    ElbowLeft = list(np.argwhere(np.abs(y - (beta0 - eps)) < on_margin_tolerance )[0])
    # Assign the new ElbowLeft and ElbowRight points
    for item in ElbowRight:
        del Center[(np.argwhere(Center == item))[0][0]]
    for item in ElbowLeft:
        del Center[(np.argwhere(Center == item))[0][0]]
    
    alphas[ElbowRight] = 0
    gammas[ElbowLeft] = 0
    alpha_er = alphas[ElbowRight]
    gamma_el = gammas[ElbowLeft]
    
    if DebugPrint:
        print('ElbowR:',ElbowRight,'ys:',y[ElbowRight],'alphas:',alpha_er)
        print('ElbowL:',ElbowLeft,'ys:',y[ElbowLeft],'gammas:',gamma_el)
        print('Center:',Center,'ys:',y[Center])
    if DebugPrint:
        print('beta0: %.20f, eps: %.20f' % ((beta0,eps)))
      
    Kstar = np.zeros(shape=[1,1], dtype=globaltype)
    Kstar = updateKstar(K, Kstar, Lambda, [], [], ElbowRight, ElbowLeft);
    f0 = np.repeat(beta0,len(K))
    
    return eps, Kstar, f0, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, mae;
    
def empty_elbows(y, eps, fl, K, Kstar, Lambda, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0,iteration):
    y_high = np.min(y - fl)
    y_low = np.max(y - fl)
    highidx = 0
    lowidx = 0
    for pt in Center:
        if y[pt] - fl[pt] > y_high:
            y_high = y[pt] - fl[pt]
            highidx = pt
        if y[pt] - fl[pt] < y_low  :
            y_low = y[pt] - fl[pt]
            lowidx = pt
    eps_new = (y_high - y_low) / 2
    
    if PlotFiguresPerIteration:
        plt.figure(iteration)
        plt.subplot(2,1,1)
        plt.plot(fl,c='blue')
        plt.plot(fl+eps,'b--',)
        plt.plot(fl-eps,'b--',)
    v0 = (y_high + y_low) / 2
    beta0_new = beta0 + v0
    if DebugPrint:
        print('beta0_old:',beta0,'beta0_new:',beta0_new)
    fl += v0
    if PlotFiguresPerIteration:
        plt.plot(fl,c='green')
        plt.plot(fl+eps_new,'g--')
        plt.plot(fl-eps_new,'g--')
        plt.plot(beta0,'g--')
        plt.scatter(np.arange(len(y)),y,c='magenta',s=2)
        plt.ylim(np.min(y)-1)
        plt.show()
    if DebugPrint:
        print('new yhigh and ylow:',y_high,y_low,',eps',eps,'->',eps_new,'idces:',lowidx,highidx)
    
    newCenter = Center.copy()
    newElbowRight = ElbowRight.copy()
    pointidx = np.argwhere(Center == highidx)
    if DebugPrint:
        print(highidx,'C->ER')
    del newCenter[pointidx[0][0]]
    newElbowRight.append(highidx)

    newElbowLeft = ElbowLeft.copy()
    pointidx = np.argwhere(Center == lowidx)
    if DebugPrint:
        print(lowidx,'C->EL')
    del newCenter[pointidx[0][0]]
    newElbowLeft.append(lowidx)
    
#    Kstar = updateKstar(K, Kstar, Lambda, ElbowRight, ElbowLeft, newElbowRight, newElbowLeft)
    Kstar = updateKstar(K, np.zeros(shape=[1,1]), Lambda, [], [], newElbowRight, newElbowLeft)
    
    if DebugPrint:
        print('new sets L',LeftRegion,'\nEL',newElbowLeft,'\nC',newCenter,'\nER',newElbowRight,'\nR',RightRegion)
    return eps_new, fl, Kstar, LeftRegion, newElbowLeft, newCenter, newElbowRight, RightRegion, alphas, gammas, beta0_new

# Exactly one of the elbows is empty.
def empty_elbow(y, eps, fl, K, Kstar, Lambda, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, iteration):
    if DebugPrint:
        print('empty_elbow: 1 elbow empty.')
    y_high = np.min(y - fl)
    y_low = np.max(y - fl)
    highidx = 0
    lowidx = 0
    for pt in Center:
        if y[pt] - fl[pt] > y_high:
            y_high = y[pt] - fl[pt]
            highidx = pt
        if y[pt] - fl[pt] < y_low  :
            y_low = y[pt] - fl[pt]
            lowidx = pt
    if len(ElbowLeft) > 0:
        y_low = y[ElbowLeft[0]] - fl[ElbowLeft[0]]
        print('y_low:',y_low)
    if len(ElbowRight) > 0:
        y_high = y[ElbowRight[0]] - fl[ElbowRight[0]]
        print('y_high:',y_high)
    eps_new = (y_high - y_low) / 2
    if DebugPrint:
        print('eps',eps,'->',eps_new)
    
    if PlotFiguresPerIteration:
        plt.figure(iteration)
        plt.subplot(2,1,1)
        plt.plot(fl,c='blue')
        plt.plot(fl+eps,'b--',)
        plt.plot(fl-eps,'b--',)
    fl += (y_high + y_low) / 2
    if PlotFiguresPerIteration:
        plt.plot(fl,c='green')
        plt.plot(fl+eps_new,'g--')
        plt.plot(fl-eps_new,'g--')
        plt.plot(beta0,'g--')
        plt.scatter(np.arange(len(y)),y,c='magenta',s=2)
        plt.ylim(np.min(y)-1)
        plt.show()
    if DebugPrint:
        print('new yhigh and ylow:',y_high,y_low,',eps',eps,'->',eps_new,'idces:',lowidx,highidx)
    
    newCenter = Center.copy()
    newElbowRight = ElbowRight.copy()
    newElbowLeft = ElbowLeft.copy()
    if len(ElbowRight) == 0:
        pointidx = np.argwhere(Center == highidx)
    if DebugPrint:
        print(highidx,'C->ER')
        del newCenter[pointidx[0][0]]
        newElbowRight.append(highidx)

    if len(ElbowLeft) == 0:
        pointidx = np.argwhere(Center == lowidx)
        print(lowidx,'C->EL')
        del newCenter[pointidx[0][0]]
        newElbowLeft.append(lowidx)
    
#    Kstar = updateKstar(K, Kstar, Lambda, ElbowRight, ElbowLeft, newElbowRight, newElbowLeft)
    Kstar = updateKstar(K, np.zeros(shape=[1,1]), Lambda, [], [], newElbowRight, newElbowLeft)
    
    if DebugPrint:
        print('new sets L',LeftRegion,'\nEL',newElbowLeft,'\nC',newCenter,'\nER',newElbowRight,'\nR',RightRegion)
    return eps_new, fl, Kstar, LeftRegion, newElbowLeft, newCenter, newElbowRight, RightRegion, alphas, gammas, beta0
    
def updateFl(fl, hl, eps, eps_new):
    if DebugPrint:
        print('\nupdateFl')
    fl_new = fl + (eps - eps_new) * hl
    if DebugPrint:
        print('fl:',fl,'\nfl_new:',fl_new,'fl_new-fl')
    return fl_new
    
def updateFl2(fl, Lambda, K, alphas, gammas, beta0, beta0_old):
    fl_new = fl
    for idx in range(0,len(fl)):
        first_term = K[81,idx] / Lambda * alphas[81]
        second_term = K[0,idx] / Lambda * gammas[0]
        third_term = beta0 - beta0_old
        if idx == 0 or idx == 81:
            change = first_term - second_term + third_term;
            if DebugPrint:
                print('new fl:',idx,first_term,second_term,third_term,change,fl[idx])
        fl_new[idx] += first_term - second_term + third_term;
    return fl_new
    
def updateAlphasGammasBeta0(b, eps, eps_new, beta0, alphas, gammas, ElbowRight, ElbowLeft):
    if DebugPrint:
        print('\nupdateAlphasGammasBeta0')
        print('old alphas gammas beta0:',alphas,'\n',gammas,'\n',beta0)
    for idx, point in enumerate(ElbowRight):
        alphas[point] = alphas[point] + (eps - eps_new) * b[1+idx]
    for idx, point in enumerate(ElbowLeft):
        gammas[point] = gammas[point] - (eps - eps_new) * b[len(ElbowRight)+1+idx]
    beta0_new = beta0 + (eps - eps_new) * b[0]
    if DebugPrint:
        print('new alphas:',alphas,'\ngammas',gammas,'\nbeta',beta0,'->',beta0_new)
    return alphas, gammas, beta0_new

def registerEvents(eps, eps_new, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, 
                                                                            eps_R_to_ER, eps_C_to_ER, eps_L_to_EL, eps_C_to_EL,
                                                                            eps_ER_to_R, eps_ER_to_C, eps_EL_to_L, eps_EL_to_C, alphas, gammas, validEventsEnter, validEventsExit):
    if DebugPrint:
        print('\nregisterEvents')
    # TODO: if exiting, do exiting only. if entering, do entering only.
    newLeftRegion = LeftRegion.copy()
    newElbowLeft = ElbowLeft.copy()
    newCenter = Center.copy()
    newElbowRight = ElbowRight.copy()
    newRightRegion = RightRegion.copy()
    newAlphas = alphas
    newGammas = gammas
    
    if eps_new in validEventsEnter:
        R_ER = np.argwhere(np.logical_and(eps_new <= eps_R_to_ER, eps_R_to_ER < eps - rho))
        C_ER = np.argwhere(np.logical_and(eps_new <= eps_C_to_ER, eps_C_to_ER < eps - rho))
        L_EL = np.argwhere(np.logical_and(eps_new <= eps_L_to_EL, eps_L_to_EL < eps - rho))
        C_EL = np.argwhere(np.logical_and(eps_new <= eps_C_to_EL, eps_C_to_EL < eps - rho))
        assert(R_ER.size <= 1)
        assert(C_ER.size <= 1)
        assert(L_EL.size <= 1)
        assert(C_EL.size <= 1)
        for event in R_ER:
            if DebugPrint:
                print(RightRegion[event[0]],'R->ER (eps needed:)',eps_R_to_ER[event[0]])
            point = RightRegion[event[0]]
            LastEventPoint = point
            LastEvent = ['R','ER']
            del newRightRegion[np.argwhere(newRightRegion == point)[0][0]]
            newElbowRight.append(point)
        for event in C_ER:
            if DebugPrint:
                print(Center[event[0]],'C->ER (eps needed:)',eps_C_to_ER[event[0]])
            point = Center[event[0]]
            LastEventPoint = point
            LastEvent = ['C','ER']
            del newCenter[np.argwhere(newCenter == point)[0][0]]
            newElbowRight.append(point)
        for event in L_EL:
            if DebugPrint:
                print(LeftRegion[event[0]],'L->EL (eps needed:)',eps_L_to_EL[event[0]])
            point = LeftRegion[event[0]]
            LastEventPoint = point
            LastEvent = ['L','EL']
            del newLeftRegion[np.argwhere(newLeftRegion == point)[0][0]]
            newElbowLeft.append(point)
        for event in C_EL:
            if DebugPrint:
                print(Center[event[0]],'C->EL (eps needed:)',eps_C_to_EL[event[0]])
            point = Center[event[0]]
            LastEventPoint = point
            LastEvent = ['C','EL']
            del newCenter[np.argwhere(newCenter == point)[0][0]]
            newElbowLeft.append(point)
    if DebugPrint:
        print('events:',R_ER,C_ER,L_EL,C_EL)
    else: # not if eps_new in validEventsEnter:
        ER_R = np.argwhere(np.logical_and(eps_new <= eps_ER_to_R, eps_ER_to_R < eps - rho))
        ER_C = np.argwhere(np.logical_and(eps_new <= eps_ER_to_C, eps_ER_to_C < eps - rho))
        EL_L = np.argwhere(np.logical_and(eps_new <= eps_EL_to_L, eps_EL_to_L < eps - rho))
        EL_C = np.argwhere(np.logical_and(eps_new <= eps_EL_to_C, eps_EL_to_C < eps - rho))  
        assert(ER_R.size <= 1)
        assert(ER_C.size <= 1)
        assert(EL_L.size <= 1)
        assert(EL_C.size <= 1)
        if DebugPrint:
            print('ELL2',EL_L,EL_C)
        for event in ER_R:
            if DebugPrint:
                print(ElbowRight[event[0]],'ER->R (eps needed:)',eps_ER_to_R[event[0]])
            point = ElbowRight[event[0]]
            LastEventPoint = point
            LastEvent = ['ER','R']
            del newElbowRight[np.argwhere(newElbowRight == point)[0][0]]
            newRightRegion.append(point)
            newAlphas[point] = 1
        for event in ER_C:
            if DebugPrint:
                print(ElbowRight[event[0]],'ER->C (eps needed:)',eps_ER_to_C[event[0]])
            point = ElbowRight[event[0]]
            LastEventPoint = point
            LastEvent = ['ER','C']
            del newElbowRight[np.argwhere(newElbowRight == point)[0][0]]
            newCenter.append(point)
            newAlphas[point] = 0
        for event in EL_L:
            if DebugPrint:
                print(ElbowLeft[event[0]],'EL->L (eps needed:)',eps_EL_to_L[event[0]])
            point = ElbowLeft[event[0]]
            LastEventPoint = point
            LastEvent = ['EL','L']
            del newElbowLeft[np.argwhere(newElbowLeft == point)[0][0]]
            newLeftRegion.append(point)
            newGammas[point] = 1
        for event in EL_C:
            if DebugPrint:
                print(ElbowLeft[event[0]],'EL->C (eps needed:)',eps_EL_to_C[event[0]])
            point = ElbowLeft[event[0]]
            LastEventPoint = point
            LastEvent = ['EL','C']
            del newElbowLeft[np.argwhere(newElbowLeft == point)[0][0]]
            newCenter.append(point)
            newGammas[point] = 0
        
        if DebugPrint:
            print('events:',ER_R,ER_C,EL_L,EL_C)
    
    if DebugPrint:
        print('new sets L',newLeftRegion,'\nEL',newElbowLeft,'\nC',newCenter,'\nER',newElbowRight,'\nR',newRightRegion)
    return newLeftRegion, newElbowLeft, newCenter, newElbowRight, newRightRegion, newAlphas, newGammas, LastEventPoint, LastEvent
    
def find_b(eps, Kstar, ElbowLeft, ElbowRight):
    if DebugPrint:
        print('\nfind_b')
    n = Kstar.shape[0];
    yA = np.asarray([0] + [1] * len(ElbowRight) + [-1] * len(ElbowLeft));
    assert(n == len(yA));
    if DebugPrint:
        print('yA',yA)
    bA = np.dot(np.linalg.inv(Kstar),yA)
    b = np.linalg.solve(Kstar, yA)
    if DebugPrint:
        print('b should be equal to bA?')
        print('b:',b)
        print('ba:',bA)
    return b
    
def find_h(K, Kstar, b, Lambda, ElbowLeft, ElbowRight):
    if DebugPrint:
        print('\nfind_h')
    # Ker, Kel: extract just the rows of K for EL and ER
    Ker = np.ndarray(shape=[len(K), len(ElbowRight)], dtype=globaltype)
    Kel = np.ndarray(shape=[len(K), len(ElbowLeft)], dtype=globaltype)
    hl = np.ndarray(shape=[len(K)], dtype=globaltype)
    bEr = np.ndarray(shape=[len(ElbowRight)], dtype=globaltype)
    bEl = np.ndarray(shape=[len(ElbowLeft)], dtype=globaltype)
    for idx, point in enumerate(ElbowRight):
        Ker[:,idx] = K[point,:]
        bEr[idx] = b[idx+1]
#        print('idx point right:',idx,point,bEr[idx])
    for idx, point in enumerate(ElbowLeft):
        Kel[:,idx] = K[point,:]
        
        bEl[idx] = b[idx+1+len(ElbowRight)]
#        print('idx point left:',idx,point,bEl[idx])
    if DebugPrint:
        print('Ker, Kel:',Ker,Kel,Ker.shape,Kel.shape)
        print('bEr, bEl:',bEr,bEl,bEr.shape,bEl.shape)
    
    for pointIdx in np.arange(len(K)):
        first_term = np.dot( Ker[pointIdx], bEr) / Lambda
        second_term = np.dot( Kel[pointIdx], bEl) / Lambda
        third_term = b[0]
        hl[pointIdx] = first_term + second_term + third_term
    return hl;
    
def exit_elbows_events(eps, alphas, gammas, Lambda, b, ElbowLeft, ElbowRight):
    if DebugPrint:
        print('\nexit_elbows_events')
    AlphasER = np.asarray([alphas[item] for item in ElbowRight])
    GammasEL = np.asarray([gammas[item] for item in ElbowLeft])
    bER = b[1:1+len(ElbowRight)]
    bEL = b[1+len(ElbowRight):]
    if DebugPrint:
        print('Alphas_ER:',AlphasER,',Gammas_EL:',GammasEL)
        print('bER,bEL:',bER,bEL)

    eps_ER_to_R = eps + (AlphasER - 1) / bER
    eps_ER_to_C = eps + AlphasER / bER
    eps_EL_to_L = eps - (GammasEL - 1) / bEL
    eps_EL_to_C = eps - GammasEL / bEL
    if DebugPrint:
        print('ER -> R:',eps_ER_to_R)
        print('ER -> C:',eps_ER_to_C)
        print('EL -> L:',eps_EL_to_L)
        print('EL -> C:',eps_EL_to_C)
    return eps_ER_to_R, eps_ER_to_C, eps_EL_to_L, eps_EL_to_C;
    
def enter_elbows_events(fl, eps, y, hl, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion):
    if DebugPrint:
        print('\nenter_elbows_events')
    eps_R_to_ER = np.ndarray(shape=[len(RightRegion)], dtype=globaltype)
    eps_C_to_ER = np.ndarray(shape=[len(Center)], dtype=globaltype)
    eps_C_to_EL = np.ndarray(shape=[len(Center)], dtype=globaltype)
    eps_L_to_EL = np.ndarray(shape=[len(LeftRegion)], dtype=globaltype)
    for idx, item in enumerate(RightRegion):
        eps_R_to_ER[idx] = (fl[item] + eps*hl[item] - y[item]) / (hl[item] - 1)
    for idx, item in enumerate(LeftRegion):
        eps_L_to_EL[idx] = (fl[item] + eps*hl[item] - y[item]) / (hl[item] + 1)
    for idx, item in enumerate(Center):
        eps_C_to_ER[idx] = (fl[item] + eps*hl[item] - y[item]) / (hl[item] - 1)
        eps_C_to_EL[idx] = (fl[item] + eps*hl[item] - y[item]) / (hl[item] + 1)
    if DebugPrint:
        print('eps_R_to_ER, eps_C_to_ER, eps_L_to_EL, eps_C_to_EL:',eps_R_to_ER, eps_C_to_ER, eps_L_to_EL, eps_C_to_EL)
    return eps_R_to_ER, eps_C_to_ER, eps_L_to_EL, eps_C_to_EL
    
def calcStats(y, fl, eps):
    errors = fl - y
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
    
def test_convergence_correctness(x,y,fl, Lambda, eps, Gamma, alphas, gammas, ElbowLeft, ElbowRight):
    svr = sklearn.svm.SVR(gamma=Gamma, epsilon=eps, C = 1 / Lambda, tol=1e-10)
    res = svr.fit(x,y)
    output = fl - res.predict(x)
#    if PlotFiguresPerIteration:
    plt.subplot(2,1,2)
    plt.plot(np.log(np.abs(output))/np.log(10))
    # Compare alphas - gammas.
    if DebugPrint:
        print('nSupport vectors:',len(ElbowLeft)+len(ElbowRight),len(res.dual_coef_[0]))
        print('support vectors ER:',alphas[ElbowRight],res.dual_coef_[0][res.dual_coef_[0] > 0])
        print('support vectors EL:',gammas[ElbowLeft],res.dual_coef_[0][res.dual_coef_[0] < 0])
    if DebugPrint:
        print('difference from the svr',list(output))

def combined_svrpath(x,y,K,Lambda,maxIterations, lambdamin, rho, RBFGamma):
    moveIdx = 0 
    start = time.time()
    eps, Kstar, fl, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, error = svrinit(K,y,Lambda);
    LastEventPoint = None
    LastEvent = None
    res = { 'alphas' : [], 'gammas' : [], 'beta0' : [], 'maes' : [], 'errors' : [], 
                    'eps' : [], 'ElbowLeft' : [], 'ElbowRight' : [], 'Center' : [], 'RightRegion' : [], 'LeftRegion' : [], 'TimeTaken' : float('Inf') }
    alphas_svr_update = []
    gammas_svr_update = []
    beta0_svr_update = []
    lambda_svr_update = []
    ElbowLeft_svr_update = [] 
    ElbowRight_svr_update = [] 
    Center_svr_update = [] 
    RightRegion_svr_update = [] 
    LeftRegion_svr_update = []

#    test_convergence_correctness(x,y,fl, Lambda, eps, Gamma=RBFGamma)
    while moveIdx < maxIterations:
        moveIdx += 1;
        #print('\n\n================= Iteration',moveIdx,'===================')
        if (len(ElbowLeft) == 0 and len(ElbowRight) == 0) or (1 == 1):
            print('Not here.')
            eps, fl, Kstar, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0 = empty_elbows(y, eps, fl, K, Kstar, Lambda, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, moveIdx)
        elif len(ElbowLeft) > 0 and len(ElbowRight) > 0:
            #if (moveIdx % 2 != 0):
                #eps, fl, Kstar, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, LastEventPoint, LastEvent = update_eps(
                #     eps=eps, y=y, fl=fl, K=K, Kstar=Kstar, Lambda=Lambda, 
                #     LeftRegion=LeftRegion, ElbowLeft=ElbowLeft, Center=Center, ElbowRight=ElbowRight, RightRegion=RightRegion, 
                #     alphas=alphas, gammas=gammas, beta0=beta0, iteration=moveIdx, LastEventPoint=LastEventPoint, LastEvent=LastEvent)
                
                #backup_lambda_update = { 'eps' :eps, 'y' :y, 'fl':fl, 'K':K, 'Kstar':Kstar, 'Lambda':Lambda, 'LeftRegion':LeftRegion, 'ElbowLeft':ElbowLeft, 'Center':Center, 'ElbowRight':ElbowRight, 
               #'RightRegion':RightRegion, 'alphas':alphas.copy(), 'gammas':gammas.copy(), 'beta0':beta0, 'iteration':moveIdx, 'LastEventPoint':LastEventPoint, 'LastEvent':LastEvent }
                if(moveIdx > -10):
                    #print('Before lambda-svr update:')
                            #else:
                    alphas_svr_update, gammas_svr_update, beta0_svr_update, lambda_svr_update, ElbowLeft_svr_update, ElbowRight_svr_update, Center_svr_update, RightRegion_svr_update, LeftRegion_svr_update = lambda_path.svr_update(
                        x=x, y=y, K=K.copy(), eps=eps, lambdamin=lambdamin, rho=rho, Kstar=Kstar.copy(), Lambda=Lambda, 
                        LeftRegion=LeftRegion.copy(), ElbowLeft=ElbowLeft.copy(), Center=Center.copy(), ElbowRight=ElbowRight.copy(), RightRegion=RightRegion.copy(), 
                        alphas=alphas.copy(), gammas=gammas.copy(), beta0=beta0.copy(), iteration=moveIdx, LastEventPoint=LastEventPoint.copy(), LastEvent=LastEvent.copy())
                    #print('APLHA before lambda-svr update:',alphas)
                    #print('APLHAS after lambda-svr update:',alphas_svr_update)
                    print('SUM ALPHAS BEFORE/AFTER::', sum(alphas), sum(alphas_svr_update))
                    if(sum(alphas_svr_update) > 0.000001):
                        print('ACCEPT LAMBDA UPDATE!!')
                        alphas = alphas_svr_update
                        gammas = gammas_svr_update
                        beta0 = beta0_svr_update
                        Lambda = lambda_svr_update
                        ElbowLeft = ElbowLeft_svr_update
                        ElbowRight = ElbowRight_svr_update
                        Center = Center_svr_update
                        RightRegion = RightRegion_svr_update
                        LeftRegion = LeftRegion_svr_update 
                    
                #if (sum(alphas)
                ##gammas_svrpath = gammas
                ##alphas, gammas_svrpath, beta0, maes_svrpath, errors_svrpath, Lambda, ElbowLeft, ElbowRight, Center, RightRegion, LeftRegion, time_taken_svrpath = test_svrpath.svrpath(x,y,K,eps, Lambda=Lambda, 
                ##     LeftRegion=LeftRegion, ElbowLeft=ElbowLeft, Center=Center, ElbowRight=ElbowRight, RightRegion=RightRegion, alphas=alphas, gammas_svrpath=gammas_svrpath, beta0=beta0, maxIterations=10, lambdamin=lambdamin, rho = rho)


            # def svrpath(x,y,K,eps,maxIterations, lambdamin, rho):
            # return { 'alphas' : np.asarray(AllAlphas), 'gammas' : np.asarray(AllGammas), 'beta0' : np.asarray(AllBetaZeros), 'maes' : np.asarray(maes), 'errors' : np.asarray(errTerms),
            # 'lambdas' : lambdas, 'ElbowLeft' : elbowL, 'ElbowRight' : elbowR, 'Center' : centers, 'RightRegion' : rights, 'LeftRegion' : lefts, 'TimeTaken' : timeTaken }


              
                
                #return { 'alphas' : np.asarray(AllAlphas), 'gammas' : np.asarray(AllGammas), 'beta0' : np.asarray(AllBetaZeros), 'maes' : np.asarray(maes), 'errors' : np.asarray(errTerms),
            #'lambdas' : lambdas, 'ElbowLeft' : elbowL, 'ElbowRight' : elbowR, 'Center' : centers, 'RightRegion' : rights, 'LeftRegion' : lefts, 'TimeTaken' : timeTaken }
                
                #def svr_update(x,y,K,eps, lambdamin, rho, Kstar, ElbowLeft, ElbowRight):
               # Lambda, fl, Kstar, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, LastEventPoint, LastEvent = lambda_path.svr_update(
               #         x=x, y=y, K=K, eps=eps, lambdamin=lambdamin, rho=rho, Kstar=Kstar, ElbowLeft=ElbowLeft, ElbowLeft=ElbowRight)
                        
                        
                        
                    #eps=eps, y=y, fl=fl, K=K, Kstar=Kstar, Lambda=Lambda, 
                    #LeftRegion=LeftRegion, ElbowLeft=ElbowLeft, Center=Center, ElbowRight=ElbowRight, RightRegion=RightRegion, 
                    #alphas=alphas, gammas=gammas, beta0=beta0, iteration=moveIdx, LastEventPoint=LastEventPoint, LastEvent=LastEvent)
        else:
            eps, fl, Kstar, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0 = empty_elbow(y, eps, fl, K, Kstar, Lambda, LeftRegion, ElbowLeft, Center, ElbowRight, RightRegion, alphas, gammas, beta0, moveIdx)
        mae, errors = calcStats(y, fl, eps)
        res['alphas'].append(alphas)
        res['gammas'].append(gammas)
        res['beta0'].append(beta0)
        res['maes'].append(mae)
        res['errors'].append(errors)
        res['eps'].append(eps)
        res['ElbowLeft'].append(ElbowLeft)
        res['ElbowRight'].append(ElbowRight)
        res['Center'].append(Center)
        res['RightRegion'].append(RightRegion)
        res['LeftRegion'].append(LeftRegion)        
        
        if moveIdx == 34000:
            plt.figure(moveIdx)
            plt.subplot(2,1,1)
            plt.plot(fl,c='blue')
            plt.plot(fl+eps,'b--',)
            plt.plot(fl-eps,'b--',)
            plt.scatter(np.arange(0,len(y)),y, s=1)
            test_convergence_correctness(x,y,fl, Lambda, eps, Gamma=RBFGamma, alphas=alphas,gammas=gammas, ElbowLeft = ElbowLeft, ElbowRight = ElbowRight)
            end = time.time()
            m, s = divmod(start-end, 60)
            h, m = divmod(m, 60)
            timeTaken = ("%d:%02d:%02d" % (h, m, s))
            res['TimeTaken'] = timeTaken
            return res
            
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    timeTaken = ("%d:%02d:%02d" % (h, m, s))
    res['TimeTaken'] = timeTaken
    return res