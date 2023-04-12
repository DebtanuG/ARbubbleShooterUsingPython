from flask import Flask, redirect, render_template, url_for, Response
import cv2
import mediapipe as mp
import time
import random
import cvzone
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
hands = mp_hands.Hands(max_num_hands=4)
app = Flask(__name__)



cap = cv2.VideoCapture(0)

success, image1 = cap.read()
Xaxis = image1.shape[0]
Yaxis = image1.shape[1]
#print(Yaxis,Xaxis)



def trigger_block1(image,x,x1,x2,y,y1,y2,score,speed_1,pos1,up1):
    if x>=x1 and x<=x2 and y>=y1 and y<=y2:
        score=score+1
        
        speed_1 = speed_1*2
        
        pos1 = random.randint(20,Yaxis-20)
        
        up1 = Xaxis-50
        
        
    return image,score,speed_1,pos1,up1
def trigger_block2(image,x,x1,x2,y,y1,y2,score,speed_2,pos2,up2):
    if x>=x1 and x<=x2 and y>=y1 and y<=y2:
        score=score+1
        
        speed_2 = speed_2*2
        
        pos2 = random.randint(5,Yaxis-20)
        
        up2 = Xaxis-50
        
        
    return image,score,speed_2,pos2,up2
def trigger_block3(image,x,x1,x2,y,y1,y2,score,speed_3,pos3,up3):
    if x>=x1 and x<=x2 and y>=y1 and y<=y2:
        score=score+1
        
        speed_3 = speed_3*2
        
        pos3 = random.randint(5,Yaxis-20)
        
        up3 = Xaxis-50
        
    return image,score,speed_3,pos3,up3

def gen_frames():
    score = 0
    #start = 0
    #end = 0
    #flag =20
    #cap = cv2.VideoCapture(0)
    up1=0
    life=4
    speed_1 = random.randint(5,10)
    speed_2 = random.randint(5,10)
    speed_3 = random.randint(2,10)
    '''
    speed_4 = random.randint(2,5)
    speed_5 = random.randint(2,5)
    speed_6 = random.randint(2,5)
    '''
    pos1 = random.randint(20,Yaxis-20)
    pos2 = random.randint(20,Yaxis-20)
    pos3 = random.randint(20,Yaxis-20)
    '''
    pos4 = random.randint(20,Yaxis-20)
    pos5 = random.randint(20,Yaxis-20)
    pos6 = random.randint(20,Yaxis-20)
    '''
    up1,up2,up3,up4,up5,up6 = Xaxis-50,Xaxis-50,Xaxis-50,Xaxis-50,Xaxis-50,Xaxis-50
    bush = cv2.imread("grass-removebg.png",cv2.IMREAD_UNCHANGED)
    bush = bush[:250,:640,:]
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        #image[240:480,:,:]=cv2.resize(bush, (640,240))
        #image = cvzone.overlayPNG(image,bush[:250,:640,:],[0,Xaxis-167])
        sub_img1 = image[10:Xaxis//4-70, 10:Yaxis//4-40]
        white_rect1 = np.ones(sub_img1.shape, dtype=np.uint8,) * 255
        res1 = cv2.addWeighted(sub_img1, 0.6, white_rect1, 0.4, 1.0, cv2.LINE_AA)
        image[10:Xaxis//4-70, 10:Yaxis//4-40] = res1
  
        #cv2.rectangle(image,(10,10),(Yaxis//4+20,Xaxis//4-10),(0,0,0),-1,cv2.LINE_AA)
        cv2.putText(image, 'Score: {}'.format(score), (25, Xaxis//4-85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (77, 55, 26), 1,cv2.LINE_AA)
        
        sub_img2 = image[10:Xaxis//4-70, Yaxis-110:Yaxis-10]
        white_rect2 = np.ones(sub_img2.shape, dtype=np.uint8,) * 255
        res2 = cv2.addWeighted(sub_img2, 0.6, white_rect2, 0.4, 1.0, cv2.LINE_AA)
        image[10:Xaxis//4-70, Yaxis-110:Yaxis-10] = res2
        
        #cv2.rectangle(image,(Yaxis-200,10),(Yaxis-10,Xaxis//4-10),(0,0,0),-1,cv2.LINE_AA)
        cv2.putText(image, 'Life: {}'.format(life), (Yaxis-90, Xaxis//4-85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (77, 55, 26), 1,cv2.LINE_AA)
        

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(rgb)
        image.flags.writeable = True
        
        #TOP left and right
        #random_ball_pos()
        if up1<10:
                life=life-1
                speed_1 = random.randint(10,20)
                pos1 = random.randint(20,Yaxis-20)
                up1 = Xaxis-50
        if up2<10:
                life=life-1
                speed_2 = random.randint(10,20)
                pos2 = random.randint(20,Yaxis-20)
                up2 = Xaxis-50
        if up3<10:
                life=life-1
                speed_3 = random.randint(10,20)
                pos3 = random.randint(20,Yaxis-20)
                up3 = Xaxis-50   
        if life<=0:
                sub_img3 = image[Xaxis//2-50:Xaxis//2+50, Yaxis//2-200:Yaxis//2+200]
                white_rect3 = np.ones(sub_img3.shape, dtype=np.uint8,) * 0
                res3 = cv2.addWeighted(sub_img3, 0.4, white_rect3, 0.6, 1.0, cv2.LINE_AA)
                image[Xaxis//2-50:Xaxis//2+50, Yaxis//2-200:Yaxis//2+200] = res3
                cv2.putText(image, 'Game Over', (Yaxis//4-10, Xaxis//2+15), cv2.FONT_HERSHEY_PLAIN, 4, (255, 242, 238), 5,cv2.LINE_AA)
                life=0
                speed_1 = 0
                pos1 = -20
                speed_2 = 0
                pos2 = -20
                speed_3 = 0
                pos3 = -20
        
        cv2.circle(image, (pos1,up1), 20, (89, 89, 255), -1,cv2.LINE_AA)
        cv2.circle(image, (pos2,up2), 20, (89, 89, 255), -1,cv2.LINE_AA)
        cv2.circle(image, (pos3,up3), 20, (89, 89, 255), -1,cv2.LINE_AA)
        image = cvzone.overlayPNG(image,bush,[0,Xaxis-167])

 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                '''
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                '''
    
                x = int(hand_landmarks.landmark[8].x * Yaxis)
                y = int(hand_landmarks.landmark[8].y * Xaxis)
                #xt = int(hand_landmarks.landmark[7].x * image.shape[1])
                #yt = int(hand_landmarks.landmark[7].y * image.shape[0])
                #cv2.putText(image,'x:{a} y:{b} '.format(a=x,b=y),(x,y), cv2.FONT_HERSHEY_PLAIN,0.8,(255,255,255),2)
                #cv2.putText(image,'x:{a} y:{b} '.format(a=xt,b=yt),(xt,yt), cv2.FONT_HERSHEY_PLAIN,0.8,(255,255,255),2)
                cv2.circle(image, (x,y), 5, [255,255,255], -1)
                #cv2.circle(image, (xt,yt), 5, [0,255,0], -1)
                #print('hand x',x, 'ball x' ,pos1-20,pos1+20, 'hand y',y, 'ball y' ,up1-20,up1+20)
                image,score,speed_1,pos1,up1 = trigger_block1(image, x,pos1-20,pos1+20,y,up1-20,up1+20,score,speed_1,pos1,up1)
                image,score,speed_2,pos2,up2 = trigger_block2(image, x,pos2-20,pos2+20,y,up2-20,up2+20,score,speed_2,pos2,up2)
                image,score,speed_3,pos3,up3 = trigger_block3(image, x,pos3-20,pos3+20,y,up3-20,up3+20,score,speed_3,pos3,up3)
      
    
        up1=up1-speed_1
        up2=up2-speed_2
        up3=up3-speed_3     
  
  #image = cv2.flip(image, 1)
        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

        
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == "__main__":
    app.run()