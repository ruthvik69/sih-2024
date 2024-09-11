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
} from "react-native";
import {
	Appbar,
	Dialog,
	IconButton,
	MD3Colors,
	PaperProvider,
	Portal,
	Text,
} from "react-native-paper";

let cam: CameraView | null;

export default function App() {
	const [facing, setFacing] = useState<CameraType>("back");
	const [permission, requestPermission] = useCameraPermissions();
	const [photo, setPhoto] = useState<CameraCapturedPicture | undefined>();
	const [visible, setVisible] = useState(false);

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
			const res = await fetch("http://localhost:8080/detect-image", {
				method: "POST",
				body: JSON.stringify({
					image: photo.base64,
				}),
			});

			const result = await res.json();
			console.log(result);
			router.push("/result");
		} catch {
			setVisible(true);
		}
	}

	function hideDialog() {
		setVisible(false);
	}

	return (
		<>
			<Appbar.Header>
				<Appbar.Action icon="arrow-left" onPress={() => {}} />
				<Appbar.Content title="Crop Disease Detection" />
			</Appbar.Header>
			<View style={styles.container}>
				<Portal>
					<Dialog visible={visible} onDismiss={hideDialog}>
						<Dialog.Title>Alert</Dialog.Title>
						<Dialog.Content>
							<Text variant="titleMedium">
								Some Error Ocurred.
							</Text>
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
						<TouchableOpacity
							style={styles.button}
							onPress={capture}
						>
							<IconButton icon="camera" size={32} />
						</TouchableOpacity>
					</View>
				</CameraView>
				<View style={styles.imageContainer}>
					<Image
						source={photo}
						width={20}
						height={30}
						style={styles.image}
					/>
					{photo ? (
						<TouchableOpacity
							style={styles.detect}
							onPress={detect}
						>
							<IconButton
								icon="clipboard-search"
								size={20}
								iconColor="white"
							/>
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
