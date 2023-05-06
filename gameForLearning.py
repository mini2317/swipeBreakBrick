from NEAT import *
import random
from parameters import *
from sprite import *

def forward(boxes,playerPos,nn):
    def check(n):
        if type(n) == Box:
            return 0.05*n.hp
        elif type(n) == Item:
            return 0.9
        else:
            return 0
    x = (*list(map(check,sum(boxes,[]))),playerPos/SCREEN_WIDTH)
    #x = (*list(map(check,boxes)),playerPos/SCREEN_WIDTH,nowRound*0.05,ballNumber*0.05)
    return math.tanh(nn.forward(*x)[0])

def get_fitness(networks,seed):
    expandedObjectNum = EXPAND * OBJECT_NUM
    nowRound = [1 for i in range(expandedObjectNum)]
    boxes = [[[None for _ in range(WIDTH)] for _ in range(HEIGHT)] for i in range(expandedObjectNum)]
    bullets = [[] for i in range(expandedObjectNum)]
    nowShooting = [False for _ in range(expandedObjectNum)]
    playerPos = [SCREEN_WIDTH/2 for i in range(expandedObjectNum)]
    startPos = [SCREEN_HEIGHT - BULLET_SIZE/2 for i in range(expandedObjectNum)]
    delay = [0 for i in range(expandedObjectNum)]
    cnt = [0 for i in range(expandedObjectNum)]
    ballNumber = [1 for i in range(expandedObjectNum)]
    addBall = [0 for i in range(expandedObjectNum)]
    comeBackBall = [0 for i in range(expandedObjectNum)]
    theta = [0 for i in range(expandedObjectNum)]
    score = [0 for i in range(expandedObjectNum)]
    end = [False for i in range(expandedObjectNum)]
    moveToPos = [0 for i in range(expandedObjectNum)]

    def makeNewLine(boxes,nowRound,nowIdx):
        nowSeed = seed[nowIdx%EXPAND]
        boxes.pop()
        if sum(map(lambda x : type(x) == Box,boxes[-1])):
            return GAME_OVER
        newLine = [None]*WIDTH
        lineNum = set(range(WIDTH))
        random.seed(nowSeed+nowRound)
        idx = random.randint(0,WIDTH-1)
        #print(idx)
        newLine[idx] = Item()
        lineNum.remove(idx)
        lineNum = list(lineNum)
        random.seed(nowSeed+nowRound)
        random.shuffle(lineNum)
        random.seed(nowSeed+nowRound)
        cutter = lineNum[:random.randint(1,WIDTH-1)]
        for i in cutter:
            newLine[i] = Box(nowRound)
        #print(len(boxes))
        boxes.insert(0,newLine)
    
    for nowIdx in range(expandedObjectNum):
        makeNewLine(boxes[nowIdx],1,nowIdx)
    roll = 0
    while True:
        for nowIdx in range(expandedObjectNum):
            if not end[nowIdx]:
                if any(end[EXPAND*(nowIdx//EXPAND):EXPAND*(1+nowIdx//EXPAND)]):
                    end[nowIdx] = True
                network = networks[nowIdx//EXPAND]
                if (not nowShooting[nowIdx]):
                    #print(network.edges)
                    #forwardResult = forward(boxes[nowIdx],playerPos[nowIdx],ballNumber[nowIdx],nowRound[nowIdx],network)
                    forwardResult = forward(boxes[nowIdx],playerPos[nowIdx],network)
                    theta[nowIdx] = -math.radians(forwardResult*(180-2*DEGREE_LIMIT)+DEGREE_LIMIT)
                    bullets[nowIdx].append(Bullet(playerPos[nowIdx],theta[nowIdx]))
                    delay[nowIdx] = 10
                    nowShooting[nowIdx] = True
                    cnt[nowIdx] += 1
                elif nowShooting[nowIdx]:
                    if (not delay[nowIdx]):
                        if cnt[nowIdx] < ballNumber[nowIdx]:
                            bullets[nowIdx].append(Bullet(playerPos[nowIdx],theta[nowIdx]))
                            delay[nowIdx] = 10
                            cnt[nowIdx] += 1
                    else:
                        delay[nowIdx] -= 1
                popped = 0
                for i in range(len(bullets[nowIdx])):
                    idx = i - popped
                    bullet = bullets[nowIdx][idx]
                    move = bullet.move(boxes[nowIdx])
                    if move is not None:
                        box = boxes[nowIdx][move[0]][move[1]]
                        box.hp -= 1
                        score[nowIdx] += 1
                        if box.hp <= 0 :
                            if type(box) == Item:
                                addBall[nowIdx] += 1
                                score[nowIdx] += nowRound[nowIdx]-move[1]-1
                            boxes[nowIdx][move[0]][move[1]] = None
                    if bullet.y >= startPos[nowIdx]:
                        if not comeBackBall[nowIdx] : moveToPos[nowIdx] = bullet.x
                        bullets[nowIdx].pop(idx)
                        comeBackBall[nowIdx] += 1
                        if comeBackBall[nowIdx] == cnt[nowIdx]:
                            cnt[nowIdx] = 0
                            comeBackBall[nowIdx] = 0
                            playerPos[nowIdx] = moveToPos[nowIdx]
                            nowShooting[nowIdx] = False
                            ballNumber[nowIdx] += addBall[nowIdx]
                            addBall[nowIdx] = 0
                            nowRound[nowIdx] += 1
                            #print("DROP!")
                            if makeNewLine(boxes[nowIdx],nowRound[nowIdx],nowIdx) == GAME_OVER:
                                end[nowIdx] = True
                            if nowRound[nowIdx] == 50:
                                end[nowIdx] = True
                            if sum(map(lambda x : type(x) == Item,boxes[nowIdx][-1])):
                                ballNumber[nowIdx] += 1
                                boxes[nowIdx][-1] = [None]*WIDTH
                        popped += 1
        if all(end):break
        roll += 1
        #print("MAX SCORE :",max(score),"MAX ROUND :",max(nowRound),"DIED PLAYERS :",sum(end)//EXPAND)
    return [sum(score[i:i+EXPAND]) for i in range(0,len(score),EXPAND)]
        #print(score)
    return score#[score[i] for i in range(0,len(score))]