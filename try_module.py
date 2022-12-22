import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image,ImageTk
from functools import partial
import cv2
import json
from pathlib import Path
import numpy as np
from os import path, listdir
from PIL import Image
from prettytable import PrettyTable
from image_recognize import recognize_attendence
from tkinter import messagebox
from datetime import date

global take_images
Path("student_images").mkdir(exist_ok=True)
NUMBER_OF_SAMPLES = 100
STUDENT_DETAILS_FILE_PATH = "student_details.json"

if not Path(STUDENT_DETAILS_FILE_PATH).exists():
    json.dump({}, open("student_details.json", "w"))

def take_image(v1,v2):
    json_config = json.load(open("student_details.json", "r"))
    id = (v1.get())
    name = (v2.get())
    with open("student_details.json") as f:
                data =json.load(f)
    
    if (id=="" or name==""):
         messagebox.showinfo("showinfo", "ALL FIELDS TO BE FILLED")

    else:
        if(id  in data.keys()):
            messagebox.showinfo("showinfo", "ID ALREADY PRESENT")
        else:
            json_config[id] = {
                "id": id,
                "name": name
            }
            json.dump(json_config, open("student_details.json", "w"))
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            sample_num = 0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray, 1.3, 5,minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                for(x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
                    cv2.imwrite(
                        f"student_images/{id}---{sample_num}---{name}.jpg", gray[y:y+h, x:x+w])
                    cv2.imshow('frame', img)
                    sample_num += 1
                if cv2.waitKey(1) == 27:
                    break
                elif sample_num > NUMBER_OF_SAMPLES:
                    break
            cam.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("showinfo", "Training the newly loaded images")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            output = [[], []]
            for imagePath in listdir("student_images"):
                imagePath = path.join("student_images", imagePath)
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                uid = path.basename(imagePath).split("---")[0]
                if uid in json_config.keys():
                    output[0].append(imageNp)
                    output[1].append(int(uid))
            recognizer.train(output[0], np.array(output[1]))
            recognizer.save("student_images_trained.yml")
            messagebox.showinfo("showinfo","Added successfully")

    var1.set("")
    var2.set("")
    

def show(text12):
    def exit2():
        show_.destroy()
    BG_GRAY = "#ABB2B9"
    show_= Toplevel(root,width=500,height=500)
    show_.resizable(width=False,height=False)
    label = Label(show_,text="Attendance sheet",font=("Algerian",14))
    label.place(relwidth=1)
    text_box = Text(show_,width=20,height=2,font=("Arial",12))
    text_box.place(relwidth =1,relheight=0.75,rely=0.1)
    text_box.insert(END,text12)
    text_box.config(cursor="arrow",state=DISABLED)
    button = Button(show_,text="close",bg=BG_GRAY,command = exit2)
    button.place(rely = 0.85,relx=0.45)
    show_.mainloop()

    
def data():
    attendance_json = json.load(open(str(date.today())+"student_attendence.json", "r"))
    info = PrettyTable()
    info.field_names = ["id", "name", "datetime"]
    info.add_rows(attendance_json[::-1])
    show(info.get_string())

def exit1():
    root.destroy()

root=tk.Tk()
root.geometry("1500x800")
root.title("Attendence system")

var1=tk.StringVar()
var2=tk.StringVar()

img=Image.open(r"D:\kani\mini_projects\face_recognition\back_ground.png")
img=img.resize((1450,825),Image.ANTIALIAS)
photoimg=ImageTk.PhotoImage(img)
bg_img=Label(root,image=photoimg)
bg_img.place(x=25,y=25,width=1300,height=700)

main_frame=Frame(bg_img,bd=2,bg="white")
main_frame.place(x=10,y=10,width=1250,height=650)

left_frame=LabelFrame(main_frame,bd=2,bg="#ABB2B9",relief=RIDGE,text="Options available",font=("arial",12,"bold"))
left_frame.place(x=10,y=10,width=600,height=600)

right_frame=LabelFrame(main_frame,bd=2,bg="#ABB2B9",relief=RIDGE,text="Add details",font=("arial",12,"bold"))
right_frame.place(x=625,y=10,width=600,height=600)

img1=Image.open(r"D:\kani\mini_projects\face_recognition\imagerec.png")
img1=img1.resize((150,150),Image.ANTIALIAS)
photoimg1=ImageTk.PhotoImage(img1)
I1=Label(left_frame,image=photoimg1)
I1.place(x=25,y=25,width=150,height=150)

img2=Image.open(r"D:\kani\mini_projects\face_recognition\att.png")
img2=img2.resize((150,150),Image.ANTIALIAS)
photoimg2=ImageTk.PhotoImage(img2)
I2=Label(left_frame,image=photoimg2)
I2.place(x=310,y=25,width=150,height=150)

img3=Image.open(r"D:\kani\mini_projects\face_recognition\im.png")
img3=img3.resize((150,150),Image.ANTIALIAS)
photoimg3=ImageTk.PhotoImage(img3)
I3=Label(left_frame,image=photoimg3)
I3.place(x=155,y=300,width=150,height=150)

take_btn=Button(left_frame,text="Take\nattendence",command=recognize_attendence,font=("arial",12),bg="black",fg="#00ffff")
take_btn.place(x=25,y=180,width=150,height=50)

show_btn=Button(left_frame,text="Show\nattendence",command=data,font=("arial",12),bg="black",fg="#00ffff")
show_btn.place(x=310,y=180,width=150,height=50)

exit_btn=Button(left_frame,text="Exit",command=exit1,font=("arial",12),bg="black",fg="#00ffff")
exit_btn.place(x=155,y=455,width=150,height=50)

img4=Image.open(r"D:\kani\mini_projects\face_recognition\add.png")
img4=img4.resize((200,200),Image.ANTIALIAS)
photoimg4=ImageTk.PhotoImage(img4)
I4=Label(right_frame,image=photoimg4)
I4.place(x=25,y=100,width=200,height=200)

labelNum1 = Label(right_frame, text="Student\n id",font=("arial",12),bg="black",fg="#00ffff")
labelNum1.place(x=230,y=100,width=100,height=50)

labelNum2 = Label(right_frame, text="Student\n name",font=("arial",12),bg="black",fg="#00ffff")
labelNum2.place(x=230,y=175,width=100,height=50)

entryNum1 = Entry(right_frame, textvariable=var1,font=("arial",12))
entryNum1.place(x=345,y=100,width=200,height=50)

entryNum2= Entry(right_frame, textvariable=var2,font=("arial",12))
entryNum2.place(x=345,y=175,width=200,height=50)

take_images=partial(take_image,var1,var2)

add_btn=Button(right_frame,text="Add",command=take_images,font=("arial",12),bg="black",fg="#00ffff")
add_btn.place(x=345,y=255,width=150,height=50)


msg=Label(right_frame,text = "1.To add student details enter student id\n 2.Press add button",bg="black",fg="#00ffff",font=("arial",10))
msg.place(x=25,y=25,width=400,height=50)


root.mainloop()
