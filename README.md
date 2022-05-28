# Python-based measurement control system for the room-temperature scanning NV setup
## G22 @ LISE (Yacoby Lab, Harvard University)

The scripts and instruments should be used with pylabcontrol (https://github.com/LISE-B26/pylabcontrol), a generic laboratory control platform for controlling scientific equipment. 

### This setup combines a confocal scanning microscope and an atomic atomic force microscope (AFM). 
- Quantum sensing is performed on the Quantum Orchestration Platform (QOP) and AFM scanning is controlled by analog output from NI DAQs. 

### The following instruments are used on this setup:

- PulseBlaster: Programmable TTL Pulse Generator<br>
- Keysight RF generator N9310A <br>
- ROHDE & SCHWARZ RF generator SMB100A <br>
- ROHDE & SCHWARZ RF generator SGS100A <br>
- Newport XPS motion controller <br>
- Yokogawa GS200 DC Voltage / Current Source <br>
- Keysight 33120A Function/Arbitrary Waveform Generator <br>
- NI PCI-6711 DAQ: 16‑Bit, 8 Channels, 1 MS/s Analog Output Device <br>
- NI PCI-6602 DAQ: 5 V, 8-Channel Counter/Timer Device <br>
- NI PCI-6220 DAQ: 16 AI (16-Bit, 250 kS/s), 24 DIO PCI Multifunction I/O Device <br>
- NI USB-6210 DAQ: 16 AI (16-Bit, 250 kS/s), 4 DI, 4 DO USB Multifunction I/O Device <br>
- Quantum Machines Quantum Orchestration Platform (QOP):<br>
      QOP and pylabcontrol are integrated as follows. Definition, simulation and execution of QUA programs are written as part of the Script module in pylabcontrol. Parameters for QUA programs such as pulse duration, repetition number and sweep array, as well as setting parameters for other instruments, are all accessible from the pylabcontrol GUI. Data is fetched in real time through stream processing by the QM server, and visualized in GUI after simple data analysis such as normalization and fitting. In-depth data analysis can be done afterward separately.


<img src="https://user-images.githubusercontent.com/29555981/170805568-e9b84cbe-211c-4cb9-97d7-f28d388d6e57.png" width=60% height=60%>


