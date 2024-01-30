import random
import numpy as np
import tf
from tensorflow.keras import layers, models
import tensorflow as t


class Agent:
    def __init__(self, state_size, action_size, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma
        self.model = self._build_model()
        self.target_model = self._build_model()  # Aggiunto il modello target

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        state_tensor = t.convert_to_tensor([state], dtype=t.float32)

        # Ottieni i valori Q previsti dal modello
        q_values = self.model.predict(state_tensor)

        # Applica softmax per ottenere una distribuzione di probabilità
        action_probabilities = t.nn.softmax(q_values[0])

        # Campiona un'azione dalla distribuzione di probabilità
        action = t.random.categorical(t.math.log(action_probabilities), 1).numpy()[0, 0]

        print(action)

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        inputs = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Calcola il valore Q atteso utilizzando la formula di aggiornamento Q-learning
                target = reward + self.gamma * np.amax(self.target_model.predict(np.array([next_state]))[0])

            # Ottieni le stime Q attuali del modello
            current_q_values = self.model.predict(np.array([state]))[0]

            # Aggiorna il valore Q per l'azione intrapresa
            current_q_values[action] = target

            # Aggiungi gli input (stati) e i target all'array
            inputs.append(state)
            targets.append(current_q_values)

        # Converti gli array in formato numpy
        inputs = np.array(inputs)
        targets = np.array(targets)

        # Addestra il modello utilizzando il minibatch
        self.model.fit(inputs, targets, epochs=1, verbose=0)
