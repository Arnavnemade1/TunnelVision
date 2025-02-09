let device;
let characteristic;

document.getElementById('connect').addEventListener('click', async () => {
    try {
        // Specifically request Classic Bluetooth
        device = await navigator.bluetooth.requestDevice({
            // Use acceptAllDevices to see all available devices including BT Classic
            acceptAllDevices: true
        });
        
        log('Device selected: ' + device.name);
        
        const server = await device.gatt.connect();
        log('Connected to GATT server');
        
        // Log all services to see what's available
        const services = await server.getPrimaryServices();
        log('Found ' + services.length + ' services');
        
        // Try to find the Serial Port Service
        for (const service of services) {
            log('Service found: ' + service.uuid);
            try {
                const characteristics = await service.getCharacteristics();
                for (const char of characteristics) {
                    log('Characteristic: ' + char.uuid);
                    characteristic = char; // Store the first writable characteristic
                }
            } catch (e) {
                log('Error getting characteristics: ' + e);
            }
        }
        
        if (characteristic) {
            log('Connected successfully!');
        }

    } catch (error) {
        log('Error: ' + error);
    }
});
