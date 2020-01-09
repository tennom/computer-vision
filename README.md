### Movement detection in videos; log and live view of the detections
-------------------------------------------------
This is a collaboration with Bradley Aaron. BA contributed the ideas and code for movement detection. 

Do you want to log some movements from your videos by setting size of the movements and region of interest?  
Do you want to have the control of the noise level and time-coded log?  
Do you want to extend the code for live video cameras and send webhooks/emails upon detections?  
If your answer is at least one yes, then we hope that you can benefit from this code.

We will update instructions soon but if you can read the first few lines of the code, you will be able to use the code with no problem.  
If you want to accelerate the detection without the live view, please uncomment the showing code like this:  
```python
    # cv2.imshow("feed", frame1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
```