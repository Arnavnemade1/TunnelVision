let device;
let characteristic;

document.getElementById('connect').addEventListener('click', async () => {
    try {
        device = await navigator.bluetooth.requestDevice({
            // Accept all devices but filter by name in the picker
            acceptAllDevices: true,
            optionalServices: ['0000ffe0-0000-1000-8000-00805f9b34fb']
        });
        
        log('Device selected: ' + device.name);
        
        const server = await device.gatt.connect();
        log('Connected to GATT server');
        
        // Let's log all available services
        const services = await server.getPrimaryServices();
        log('Available services: ' + services.length);
        for (const service of services) {
            log('Service: ' + service.uuid);
            const characteristics = await service.getCharacteristics();
            for (const char of characteristics) {
                log('Characteristic: ' + char.uuid);
            }
        }
        
        // Then try to connect to our service
        const service = await server.getPrimaryService('0000ffe0-0000-1000-8000-00805f9b34fb');
        characteristic = await service.getCharacteristic('0000ffe1-0000-1000-8000-00805f9b34fb');
        
        log('Connected successfully');

    } catch (error) {
        log('Error: ' + error);
    }
});
