
import sys
from gymnasium import utils
import string

class textRenderer:

    def __init__(self):
        self.initializedRender = False


    def initRender(self, env):
        outfile = sys.stdout
        outfile.write("Actions: "+ str(env.nameActions) + "\n")

    def render(self,env,current,lastaction,lastreward):

        if (not self.initializedRender):
            self.initRender(env)
            self.initializedRender = True

        # Print the MDP in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        desc = ["."]

        #desc.append(" \t\tr=" + str(lastreward))

        if lastaction is not None:
            outfile.write("({})\tr={}\n".format(env.nameActions[lastaction % 26],str("{:01.2f}".format(lastreward))))
        else:
            outfile.write("")
        outfile.write("".join(''.join(line) for line in desc) + "\t")

        #if mode != 'text':
        #    return outfile