import { registerRootComponent } from 'expo';
import { initExecutorch } from 'react-native-executorch/legacy';
import { ExpoResourceFetcher } from 'react-native-executorch-expo-resource-fetcher';

import App from './App';

// ExecuTorch wajib diinisialisasi sekali di titik masuk app, sebelum API lain dipakai.
initExecutorch({ resourceFetcher: ExpoResourceFetcher });

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that whether you load the app in Expo Go or in a native build,
// the environment is set up appropriately
registerRootComponent(App);