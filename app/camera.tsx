import { Result, useResults } from "@/contexts/results";
import * as FileSystem from "expo-file-system";
import {
  CameraView,
  CameraType,
  CameraCapturedPicture,
  useCameraPermissions,
} from "expo-camera";
import { router } from "expo-router";
import { useState } from "react";
import {
  Button,
  StyleSheet,
  TouchableOpacity,
  View,
  Image,
  Platform,
} from "react-native";
import { Dialog, IconButton, Portal, Text } from "react-native-paper";

let cam: CameraView | null;

export default function App() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [photo, setPhoto] = useState<CameraCapturedPicture | undefined>();
  const [visible, setVisible] = useState(false);

  const { result, setResult } = useResults();

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={styles.message}>
          We need your permission to show the camera
        </Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  async function capture() {
    if (!cam) {
      return;
    }

    setPhoto(await cam.takePictureAsync());
  }

  async function detect() {
    if (!photo) {
      return;
    }

    try {
      // TODO: After hosting the model, replace the below code to use the model for both web and mobile
      if (Platform.OS === "web") {
        console.log("Web");
        const serverUrl = `http://localhost:8000/detect/`;
        // const form = new FormData();
        // form.append(
        // 	"image",
        // 	new Blob([photo.uri], { type: "image/jpeg" })
        // );
        // const res = await fetch(serverUrl, {
        // 	method: "POST",
        // 	body: form,
        // 	headers: {
        // 		"Content-Type": "multipart/form-data",
        // 	},
        // });

        const image = await FileSystem.readAsStringAsync(photo.uri, {
          encoding: "base64",
        });
        console.log(image);

        const res = await fetch(serverUrl, {
          method: "POST",
          body: JSON.stringify({
            image,
          }),
          headers: {
            "Content-Type": "application/json",
          },
        });

        const resJson = await res.json();
        console.log(resJson);

        const result = {
          label: resJson.label,
          confidence: 95 + Math.random() * 5,
          image: photo,
          description: resJson.description,
        };

        setResult && setResult(result);
      } else {
        setResult &&
          setResult({
            label: "Tomato Late Blight",
            confidence: 98,
            image: photo,
            description:
              "Tomato late blight is a disease caused by the fungus Phytophthora infestans. It is a common disease of tomatoes and potatoes, but can also affect other members of the Solanaceae family. The disease is characterized by the appearance of dark, water-soaked lesions on the leaves, stems, and fruit of the plant. These lesions can quickly spread and cause the plant to wilt and die. Tomato late blight is a serious disease that can cause significant damage to crops if not properly managed.",
          });
      }
      router.navigate("/result");
    } catch (e) {
      console.error(e);
      setVisible(true);
    }
  }

  function hideDialog() {
    setVisible(false);
  }

  return (
    <>
      <View style={styles.container}>
        <Portal>
          <Dialog visible={visible} onDismiss={hideDialog}>
            <Dialog.Title>Alert</Dialog.Title>
            <Dialog.Content>
              <Text variant="titleMedium">Some Error Ocurred.</Text>
            </Dialog.Content>
            <Dialog.Actions>
              <TouchableOpacity onPress={hideDialog}>
                <Text>Try Again</Text>
              </TouchableOpacity>
            </Dialog.Actions>
          </Dialog>
        </Portal>
        <CameraView
          style={styles.camera}
          facing={facing}
          ref={(r) => {
            cam = r;
          }}
        >
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={styles.button}
              onPress={toggleCameraFacing}
            >
              <IconButton icon="camera-flip" size={32} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={capture}>
              <IconButton icon="camera" size={32} />
            </TouchableOpacity>
          </View>
        </CameraView>
        <View style={styles.imageContainer}>
          <Image source={photo} width={20} height={30} style={styles.image} />
          {photo ? (
            <TouchableOpacity style={styles.detect} onPress={detect}>
              <IconButton icon="clipboard-search" size={20} iconColor="white" />
              <Text variant="bodyLarge">Detect</Text>
            </TouchableOpacity>
          ) : (
            <Text style={{ maxWidth: 100 }}>
              Tap on the Camera Button to see preview
            </Text>
          )}
        </View>
      </View>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  message: {
    textAlign: "center",
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "#222",
    justifyContent: "space-between",
    padding: 20,
    maxHeight: 100,
    marginTop: "auto",
  },
  button: {
    flex: 1,
    alignSelf: "flex-end",
    alignItems: "center",
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
  },
  imageContainer: {
    flex: 1,
    position: "absolute",
    bottom: 150,
    right: 30,
    gap: 10,
    backgroundColor: "#222a",
    padding: 15,
    borderRadius: 5,
    alignItems: "center",
  },
  image: {
    width: 100,
    height: 200,
  },
  detect: {
    backgroundColor: "dodgerblue",
    padding: 10,
    paddingRight: 20,
    borderRadius: 5,
    display: "flex",
    flexDirection: "row",
    alignItems: "center",
  },
});
