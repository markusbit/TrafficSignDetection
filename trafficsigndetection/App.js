import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { Image } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { StatusBar } from 'expo-status-bar';
import logo from './assets/logo/logo_transparent.png'; 
import CameraScreen from './components/CameraScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Camera" 
        screenOptions={{
          headerShown: true, 
          title: '', 
          headerStyle: { 
            backgroundColor: 'transparent', 
            height: 80, 
          } 
        }}
      >
        <Stack.Screen 
          name="Camera" 
          component={CameraScreen} 
          options={{
            headerLeft: () => (
              <Image
                source={logo}
                style={{ width: 103, height: 60, resizeMode: 'contain' }}
              />
            ),
          }}
        />
      </Stack.Navigator>
      <StatusBar style="auto" />
    </NavigationContainer>
  );
}