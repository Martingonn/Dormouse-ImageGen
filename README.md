# Original Author 
Development was started on March 1st, 2025, by Marcin Jacek Chmiel.
# Contributors 
As of now, there are no more contributors than the original author.
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
