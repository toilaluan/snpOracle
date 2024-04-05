module.exports = {
  apps: [
    {
      name: 'validator',
      script: 'python3',
      args: './neurons/validator.py --netuid 93 --logging.debug --logging.trace --subtensor.network test --wallet.name testValidator --wallet.hotkey default'
    },
  ],
};
