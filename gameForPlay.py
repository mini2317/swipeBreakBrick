#from NEAT import *
import pygame,sys,random
from parameters import *
from sprite import *

if __name__ == "__main__":
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
    #nn = Topology(INPUT_NUM, OUTPUT_NUM)
    #edges = [Edge(0,19,1,0,False), Edge(1,21,0.03852083877859169,1,False), Edge(2,20,1,2,False), Edge(3,20,1,3,False), Edge(4,19,1,4,False), Edge(5,20,1,5,False), Edge(6,20,1,6,False), Edge(7,18,0.5721518052830037,7,False), Edge(8,20,1,8,False), Edge(9,21,1,9,False), Edge(10,19,1,10,False), Edge(11,21,1,11,False), Edge(12,19,-0.10996947072646635,12,False), Edge(13,21,0.9468966335306592,13,False), Edge(14,20,0.9586227820397488,14,False), Edge(15,18,1,15,False), Edge(16,20,1,16,False), Edge(17,23,1,17,False), Edge(13,20,0.16470923576974894,158,False), Edge(6,19,0.602285122747954,159,True), Edge(24,20,1,454,False)]
    #nn.init(*edges)
    theta = 0

    def makeNewLine(boxes,nowRound):
        boxes.pop()
        for y in boxes:
            for box in y:
                if type(box) == Box:
                    box.skinUpdate(nowRound)
        if sum(map(lambda x : type(x) == Box,boxes[-1])):
            return GAME_OVER
        newLine = [None]*WIDTH
        lineNum = set(range(WIDTH))
        idx = random.randint(0,WIDTH-1)
        newLine[idx] = Item()
        lineNum.remove(idx)
        lineNum = list(lineNum)
        random.shuffle(lineNum)
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
        showTheta = max(min((0*math.pi + math.atan2(-mouse[1]+startPos,-mouse[0] + playerPos))%(2*math.pi),math.pi*(180-DEGREE_LIMIT)/180),math.pi*DEGREE_LIMIT/180)
        pygame.draw.line(screen,BLUE,(int(moveToPos),int(startPos)),(int(moveToPos) - math.cos(showTheta)*50, int(startPos) - math.sin(showTheta)*50))
        if mouse[2] and (not nowShooting):
            theta = min(max(math.atan2(mouse[1]-startPos,mouse[0] - playerPos),-math.pi*(180-DEGREE_LIMIT)/180),-math.pi*DEGREE_LIMIT/180)
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
                    ballNumber += addBall
                    addBall = 0
                    nowRound += 1
                    if makeNewLine(boxes,nowRound) == GAME_OVER:
                        print("YOU DIE")
                        sys.exit()
                    if sum(map(lambda x : type(x) == Item,boxes[-1])):
                        ballNumber += 1
                        boxes[-1] = [None]*WIDTH
                popped += 1
            bullet.draw(screen)
        drawBox(screen,boxes)
        pygame.display.update()