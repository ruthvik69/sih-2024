import {
	CameraView,
	CameraType,
	CameraCapturedPicture,
	useCameraPermissions,
} from "expo-camera";
import { useState } from "react";
import {
	Button,
	StyleSheet,
	Text,
	TouchableOpacity,
	View,
	Image,
} from "react-native";

let cam: CameraView | null;

export default function App() {
	const [facing, setFacing] = useState<CameraType>("back");
	const [permission, requestPermission] = useCameraPermissions();
	const [photo, setPhoto] = useState<CameraCapturedPicture | undefined>();

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

	return (
		<View style={styles.container}>
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
						<Text style={styles.text}>Flip Camera</Text>
					</TouchableOpacity>
					<TouchableOpacity style={styles.button} onPress={capture}>
						<Text style={styles.text}>Capture</Text>
					</TouchableOpacity>
				</View>
			</CameraView>
			<Image source={photo} width={20} height={30} style={styles.image} />
		</View>
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
		backgroundColor: "transparent",
		margin: 64,
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
	image: {
		position: "absolute",
		bottom: 100,
		right: 20,
		width: 100,
		height: 200,
	},
});
