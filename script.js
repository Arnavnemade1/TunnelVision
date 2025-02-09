let device;
let characteristic;
const serviceUuid = '00001101-0000-1000-8000-00805f9b34fb'; // Standard Serial Port Service UUID

document.getElementById('connect').addEventListener('click', async () => {
    try {
        device = await navigator.bluetooth.requestDevice({
            filters: [{ name: 'HC-05' }],
            optionalServices: [serviceUuid]
        });
        const server = await device.gatt.connect();
        const service = await server.getPrimaryService(serviceUuid);
        characteristic = await service.getCharacteristic(serviceUuid);
        log('Connected to HC-05');
    } catch (error) {
        log('Error: ' + error);
    }
});

document.getElementById('send').addEventListener('click', async () => {
    if (characteristic) {
        const threshold = document.getElementById('threshold').value;
        const probability = document.getElementById('probability').value;
        const command = `QT:SET,${threshold},${probability}\n`;
        await characteristic.writeValue(new TextEncoder().encode(command));
        log(`Sent: ${command}`);
    } else {
        log('Not connected');
    }
});

function log(message) {
    const logDiv = document.getElementById('log');
    logDiv.innerHTML += message + '<br>';
    logDiv.scrollTop = logDiv.scrollHeight;
}
