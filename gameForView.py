from NEAT import *
import pygame,sys,random
from parameters import *
from sprite import *

if __name__ == "__main__":
    def forward(boxes,playerPos,ballNumber,nowRound,nn):
        def check(n):
            if type(n) == Box:
                return 0.05*n.hp
            elif type(n) == Item:
                return 0.9
            else:
                return 0
        x = (*list(map(check,sum(boxes,[]))),playerPos/SCREEN_WIDTH,nowRound*0.05,ballNumber*0.05)
        #x = (*list(map(check,boxes)),playerPos/SCREEN_WIDTH,nowRound*0.05,ballNumber*0.05)
        return nn.forward(*x)
    pygame.init()
    pygame.display.set_caption("SwipeBrickBreak")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    nowRound = 1
    boxes = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]
    bullets = []
    nowShooting = False
    playerPos = SCREEN_WIDTH/2
    startPos = SCREEN_HEIGHT - BULLET_SIZE/2
    delay = 0
    cnt = 0
    ballNumber = 1
    addBall = 0
    comeBackBall = 0
    moveToPos = playerPos
    nn = Topology(INPUT_NUM, OUTPUT_NUM)
    edges = [Edge(11,45,1,0,True), Edge(10,45,0.31820876699338674,324,False), Edge(46,45,1,133,True), Edge(24,46,0.7384731866626848,393,False), Edge(46,45,0.49999960326743736,298,False), Edge(48,45,0.5648062217784794,431,True), Edge(13,46,0.2679762511178482,404,False), Edge(46,45,0.9246861853044169,85,True), Edge(47,45,-0.2412986211688054,378,False), Edge(49,48,0.5131900545513307,484,False), Edge(49,45,0.638459716100725,486,False), Edge(4,46,-0.0718237351744459,550,False), Edge(3,45,-0.22741311076349652,490,False), Edge(5,45,0.8636664303916188,430,False), Edge(47,45,0.022223677944136933,398,False), Edge(40,45,0.8438401438462697,497,False), Edge(25,48,0.40145043460529506,468,False), Edge(48,45,0,500,False), Edge(24,45,0.5530154782604499,469,False), Edge(42,45,0.8279842878743203,533,False), Edge(46,45,0.12507532915507025,220,True)]
    nn.init(*edges)
    theta = playerPos

    def makeNewLine(boxes,nowRound,seed=SEED):
        boxes.pop()
        for y in boxes:
            for box in y:
                if type(box) == Box:
                    box.skinUpdate(nowRound)
        if sum(map(lambda x : type(x) == Box,boxes[-1])):
            return GAME_OVER
        newLine = [None]*WIDTH
        lineNum = set(range(WIDTH))
        random.seed(seed)
        seed = [random.random() for i in range(EXPAND)][0]
        random.seed(seed+nowRound)
        idx = random.randint(0,WIDTH-1)
        newLine[idx] = Item()
        lineNum.remove(idx)
        lineNum = list(lineNum)
        random.seed(seed+nowRound)
        random.shuffle(lineNum)
        random.seed(seed+nowRound)
        for i in lineNum[:random.randint(1,WIDTH-1)]:
            newLine[i] = Box(nowRound)
        boxes.insert(0,newLine)
    
    def drawBox(screen,boxes):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if boxes[y][x] is not None:
                    boxes[y][x].draw(screen,(x,y))
    makeNewLine(boxes,1)
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        mouse, eventKeys, pressed_keys = keyboard()
        screen.fill(WHITE)
        Bullet(moveToPos,math.radians(90)).draw(screen)
        forwardResult = forward(boxes,playerPos,ballNumber,nowRound,nn)
        showTheta = max(min(math.pi-forwardResult[0]*math.pi,math.pi*(180-DEGREE_LIMIT)/180),math.pi*DEGREE_LIMIT/180)
        pygame.draw.line(screen,BLUE,(int(moveToPos),int(startPos)),(int(moveToPos) - math.cos(showTheta)*50, int(startPos) - math.sin(showTheta)*50))
        if (not nowShooting):
            theta = min(max(-forwardResult[0]*math.pi,-math.pi*(180-DEGREE_LIMIT)/180),-math.pi*DEGREE_LIMIT/180)
            print(-theta*180/math.pi)
            bullets.append(Bullet(playerPos,theta))
            delay = 10
            nowShooting = True
            cnt += 1
        elif nowShooting:
            if (not delay):
                if cnt < ballNumber:
                    bullets.append(Bullet(playerPos,theta))
                    delay = 10
                    cnt += 1
            else:
                delay -= 1
        popped = 0
        for i in range(len(bullets)):
            idx = i - popped
            bullet = bullets[idx]
            move = bullet.move(boxes)
            if move is not None:
                box = boxes[move[0]][move[1]]
                box.hp -= 1
                if type(box) == Box:
                    box.skinUpdate(nowRound)
                if box.hp <= 0 :
                    if type(box) == Item:
                        addBall += 1
                    boxes[move[0]][move[1]] = None
            if bullet.y >= startPos:
                if not comeBackBall : moveToPos = bullet.x
                bullets.pop(idx)
                comeBackBall += 1
                if comeBackBall == cnt:
                    cnt = 0
                    playerPos = moveToPos
                    comeBackBall = 0
                    nowShooting = False
                    ballNumber += addBall + sum(map(lambda x : type(x) == Item,boxes[-1]))
                    addBall = 0
                    nowRound += 1
                    if makeNewLine(boxes,nowRound) == GAME_OVER:
                        print("YOU DIE")
                        sys.exit()
                popped += 1
            bullet.draw(screen)
        drawBox(screen,boxes)
        pygame.display.update()