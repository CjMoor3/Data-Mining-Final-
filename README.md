# Data-Mining-Final-
### Group Members
- Jaron Ritter
- C.J. Moore
- Liam Andrade
- Steven Senger

### Env Set up
1. Create a venv
2. activate the env
3. pip install ipykernel
4. run `ipython kernel install --name "nameYouWantForKernel" --user`
5. Verfiy kernel has been set up in jupter lab. Run - `jupyter lab`
6. Install dependencies using the requirements.txt. Run - `pip install -r pathToReqFile/requirements.txt`

### Project Goals
- Classify images of blood cells as cancerous or not.

### Approach
- Neural Net
- Random Forest

### Notes
- If you take a look at the dependencies, there is a libary called plotly. It is a JS based interactive data vis library. I will be using it and I would recommend you look at it too (it's pretty cool)
- If you add a dependency please add it to the requirements.txt

### Work flow Guidelines
1. Please create a new branch off main to work on
    - Please keep changes smallish so they're not hard to review
2. Have other reivew your code and verify they review it before merging to main
3. If you add new packages into it, please update the requirements.txt
    - If you need to import a library that had ml algorithms in it, ping me (Jaron) because they need approval by Dr. Saquer

### General Process
1.	Determine correct preprocessing steps. 
    - Should we gray scale or find a way to use all three channels of color?
    - Should we save the processed images? (There are a lot of them)
    - Feature Selection? (We shouldn’t need it)
2.	Preprocess data and time it. Does it happen in a reasonable time?
3.	Test the models (NN vs Random Forest), and select the best
    - Do not throw away the results of the losing model. We need it to compare in the write up
    - This step should include hyperparameter tuning if applicable 
    - It says we should also repeat this for validity
4.	Interpret the results. 
    - Can we do this, and if so, what does it mean?
5.	Data Visualization
    - Confusion matrix, in our case false negatives will be very bad
    - Scatter Plots, this might lead us to try a clustering algorithm
6.	Note the issues you run into so we can talk about it in the write up
7.	If you have ideas on how to improve the results or future plans for this project, note it. We will need it for the write up. 
8. Write the write up (This is going to take the longest)
