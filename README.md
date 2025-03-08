# What is it?
Dormouse is a StableDiffusion-based AI Image generator that is completely free and does not collect any data. It can also be trained.
# Original Author 
Development was started on March 1st, 2025, by Marcin Jacek Chmiel.
# Contributors 
As of now, there are no more contributors than the original author.
If you have any problems or suggestions, contact me: *martingonn-dev@outlook.com*
# Disclaimer!!!
**Do not use the model in any illegal way!** Guardrail removal is only for advancing your creativity!
# How to use
1. Download the files from the release you need.
2. Ensure you have Python 3.11 and up for best compatibility (the code did not work for me on lower versions). If you use older python versions for other scripts, I recommend to create a virtual environment.
How to create python virtual environment:
  If you are below python 3.3, write:
    "sudo apt install python3-venv"
1.1 In Bash (Console) write "py -3.11 -m venv myenv". You can replace "myenv" with a custom name, though you will have to replace it in every command shown here.
   If the command works, nothing should show up.
   2.1 On Windows, write "myenv\Scripts\activate"
   2.2 On Linux, write "source myenv/bin/activate"
   If done correctly, you should see (myenv) at the beginning of the command line.
4. Download required dependencies from the "required.txt" file using "pip install -r required.txt"
5. Run the script, follow instructions in terminal.
   If Python tells you that you don't have required libraries installed even though you do, try running the script with "python" instead of "python3". Example: instead of "python3 dormouse10.py" do "python dormouse10.py"

# How to train
1. Before running script, create a folder with training images.
2. Make a folder with training images.
3. Make a .txt file with image descriptions in the order they are segregated in the folder. The description file should look like this:
  A small, cute European hedgehog in field of shamrocks, looking right
  Small, cute hedgehog in hand, looking left
  A tiny, young hedgehog, looking forward, transparent background
4. Follow the instructions. When pasting file paths, remove the "" signs. Input C:path/to/your/file instead of "C:path/to/yout/file"

# Future Additions
* Add a way to use model training
* Test .exe on windows without python/python libraries
* config file to implement guardrails
* a way to train off databases
* the model resizing images for training

# Downloads
![GitHub All Releases](https://img.shields.io/github/downloads/Martingonn/Dormouse-ImageGen/total)
