from FourRooms import FourRooms
import numpy as np

def main():

    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple')

    actSeq = [FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
              FourRooms.UP, FourRooms.UP, FourRooms.UP,
              FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
              FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN]

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    # Q-learning parameters
    num_epochs = 50
    learning_rate = 0.8
    discount_factor = 0.7
    epsilon = 0.5

    # Initialize Q-table
    num_actions = 4
    grid_width = 13  # Assuming a 13x13 grid
    num_states = grid_width * grid_width
    Q = np.zeros((num_states, num_actions))

    for epoch in range(num_epochs):

        numMoves = 0
        total_reward = 0
        isTerminal = False

        print("Epoch being calculated...")

        while not isTerminal:
            current_state = fourRoomsObj.getPosition()
            current_state_index = current_state[1] * grid_width + current_state[0]

            if (np.random.rand()<epsilon):
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[current_state_index])

            gridType, newPos, packages_remaining, isTerminal = fourRoomsObj.takeAction(action)
            # Uncomment line below to observe Agents moves.
            # print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[action], newPos, gTypes[gridType]))
            numMoves += 1
            next_state = newPos
            next_state_index = next_state[1] * grid_width + next_state[0]
            reward = 1 if gridType > 0 else 0

            Q[current_state_index][action] += learning_rate * (
                    reward + discount_factor * np.max(Q[next_state_index]) - Q[current_state_index][action])

            total_reward += reward

        print("Epoch: {0}, Total Reward: {1}, Total Moves: {2}".format(epoch + 1, total_reward, numMoves))
        
        # Show Path
        if (numMoves<100):
            fourRoomsObj.showPath(-1)
    
        # Reset the environment for a new epoch
        fourRoomsObj.newEpoch()

        epsilon*=0.9
    


if __name__ == "__main__":
    main()