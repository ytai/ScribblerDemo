import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.UIManager;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Main {
	private static Random random_ = new Random();
	private static final double COVERAGE = 0.2;
	private static final double GRAY_RESOLUTION = 128;

	/**
	 * @param args
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws IOException, InterruptedException {
		if (args.length != 1) {
			System.exit(1);
		}

		Mat original = Highgui.imread(args[0]);
		// Convert to gray-scale.
		Imgproc.cvtColor(original, original, Imgproc.COLOR_BGR2GRAY);
		// Initial resize, just for display purposes.
		final double scale = 600.0 / original.cols();
		Imgproc.resize(original, original, new Size(), scale, scale, Imgproc.INTER_LINEAR);

		// Dummy image, for display purposes.
		Mat out = new Mat();
		original.convertTo(out, CvType.CV_32FC1);
		Core.multiply(out, new Scalar(1.0 / 255), out);

		// Down-sample.
		final double SCALE = 0.2;
		Mat in = new Mat();
		Imgproc.resize(out, in, new Size(), SCALE, SCALE, Imgproc.INTER_LANCZOS4);

		// Generate preview.
		Imgproc.GaussianBlur(out, out, new Size(), 1 / SCALE);
		Core.subtract(Mat.ones(out.size(), CvType.CV_32FC1), out, out);
		Core.subtract(out, new Scalar(COVERAGE), out);
		Core.subtract(Mat.ones(out.size(), CvType.CV_32FC1), out, out);
		Core.multiply(out, new Scalar(255), out);
		out.convertTo(out, CvType.CV_8UC1);

		// Negative: bigger number = darker.
		Core.subtract(Mat.ones(in.size(), CvType.CV_32FC1), in, in);
		final double LINES_PER_PIXEL = 600.0 / in.cols();

		Core.multiply(in, new Scalar(LINES_PER_PIXEL * GRAY_RESOLUTION), in);
		in.convertTo(in, CvType.CV_16SC1);

		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e) {
		}
		final JFrame frame = new JFrame("Scribble");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		Container contentPane = frame.getContentPane();
		contentPane.setLayout(new BoxLayout(contentPane, BoxLayout.X_AXIS));

		contentPane.add(new JLabel(new ImageIcon(matToBufferedImage(out))));
		ImageComponent component = new ImageComponent(matToBufferedImage(out));
		contentPane.add(component);

		frame.pack();
		frame.setVisible(true);

		Point lastP = null;
		Point[] scaledLine = new Point[2];
		double residualDarkness = darkness(in) / LINES_PER_PIXEL;
		double totalLength = 0;
		int lines = 0;
		component.hideImage();
		while (residualDarkness > COVERAGE) {
			Point[] bestLine = nextLine(in, 1000, lastP);
			lastP = bestLine[1];
			scaleLine(bestLine, scaledLine, 1 / SCALE);
			List<int[]> line = new ArrayList<int[]>(2);
			line.add(new int[] { (int) scaledLine[0].x, (int) scaledLine[0].y });
			line.add(new int[] { (int) scaledLine[1].x, (int) scaledLine[1].y });
			totalLength += Math.hypot(scaledLine[0].x - scaledLine[1].x, scaledLine[0].y
					- scaledLine[1].y);
			component.addLine(line);
			++lines;
			residualDarkness = darkness(in) / LINES_PER_PIXEL;
			System.out.format("%d -- remaining darkness: %.0f%% length: %.1f\n", lines,
					100 * residualDarkness, totalLength);
		}
	}

	// private static Mat randomLines(Mat in, double scale, int numLines, int numAttemptsPerLine,
	// boolean continuous) {
	// // Core.subtract(in, new Scalar(0.5), in);
	// Core.multiply(in, new Scalar(1 / scale), in);
	//
	// Mat out = Mat.zeros(in.size(), CvType.CV_8U);
	// Imgproc.resize(in, in, new Size(0, 0), scale, scale, Imgproc.INTER_CUBIC);
	// Mat outMask = Mat.zeros(out.size(), CvType.CV_8U);
	// Point lastP = null;
	// Point[] scaledLine = new Point[2];
	// int lines = 0;
	// while (lines < numLines) {
	// Point[] bestLine = nextLine(in, numAttemptsPerLine, continuous ? lastP : null);
	// lastP = bestLine[1];
	// scaleLine(bestLine, scaledLine, 1 / scale);
	// outMask.setTo(new Scalar(0));
	// Core.line(outMask, scaledLine[0], scaledLine[1], new Scalar(1));
	// out.setTo(new Scalar(1), outMask);
	//
	// ++lines;
	// System.out.format("%d -- remaining: %f\n", lines, Core.sumElems(in).val[0]);
	// }
	//
	// Imgproc.threshold(out, out, 0.5, 255, Imgproc.THRESH_BINARY_INV);
	//
	// Core.normalize(in, in, 0, 255, Core.NORM_MINMAX);
	// return out;
	// }

	private static double darkness(Mat in) {
		double total = Core.sumElems(in).val[0];
		return total / in.cols() / in.rows() / 128;
	}

	/**
	 * Gets the best of several random lines.
	 * 
	 * The number of candidates is determined by the numAttempts argument. The criterion for
	 * determining the winner is the one which covers the highest average darkness in the image. As
	 * a side-effect, the winner will be subtracted from the image.
	 * 
	 * @param image
	 *            The image to approximate. Expected to be of floating point format, with higher
	 *            values representing darker areas. Should be scaled such that subtracting a value
	 *            of GRAY_RESOLUTION from a pixel corresponds to how much darkness a line going
	 *            through it adds. When the method returns, the winning line will be subtracted from
	 *            this image.
	 * @param numAttempts
	 *            How many candidates to examine.
	 * @param startPoint
	 *            Possibly, force the line to start at a certain point. In case of null, the line
	 *            will comprise two random point.
	 * @return The optimal line.
	 */
	private static Point[] nextLine(Mat image, int numAttempts, Point startPoint) {
		Mat mask = Mat.zeros(image.size(), CvType.CV_8U);
		Mat bestMask = Mat.zeros(image.size(), CvType.CV_8U);
		Point[] line = new Point[2];
		Point[] bestLine = null;
		double bestScore = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < numAttempts; ++i) {
			generateRandomLine(image.size(), startPoint, line);

			mask.setTo(new Scalar(0));
			Core.line(mask, line[0], line[1], new Scalar(GRAY_RESOLUTION));

			double score = Core.mean(image, mask).val[0];
			if (score > bestScore) {
				bestScore = score;
				Mat t = mask;
				mask = bestMask;
				bestMask = t;
				bestLine = line.clone();
			}
		}
		Core.subtract(image, bestMask, image, bestMask, image.type());
		return bestLine;
	}

	private static void scaleLine(Point[] line, Point[] scaledLine, double scale) {
		scaledLine[0] = new Point(line[0].x * scale, line[0].y * scale);
		scaledLine[1] = new Point(line[1].x * scale, line[1].y * scale);
	}

	private static void generateRandomLine(Size s, Point pStart, Point[] result) {
		if (pStart == null) {
			result[0] = new Point(random_.nextDouble() * s.width, random_.nextDouble() * s.height);
		} else {
			result[0] = pStart;
		}
		do {
			result[1] = new Point(random_.nextDouble() * s.width, random_.nextDouble() * s.height);
		} while (result[0].equals(result[1]));
	}

	static {
		System.loadLibrary("opencv_java246");
	}

	private static class ImageComponent extends Component {
		private static final long serialVersionUID = -8921722655371221897L;

		private final BufferedImage image_;
		private final List<List<int[]>> lines_ = new LinkedList<List<int[]>>();
		private boolean showImage_ = true;

		public ImageComponent(BufferedImage image) {
			image_ = image;
		}

		public synchronized void hideImage() {
			showImage_ = false;
			repaint();
		}

		public synchronized void addLine(List<int[]> line) {
			lines_.add(line);
			repaint();
		}

		@Override
		public Dimension getPreferredSize() {
			return new Dimension(image_.getWidth(), image_.getHeight());
		}

		@Override
		public synchronized void paint(Graphics g) {
			if (showImage_) {
				g.drawImage(image_, 0, 0, image_.getWidth(), image_.getHeight(), 0, 0,
						image_.getWidth(), image_.getHeight(), null);
			} else {
				g.setColor(Color.WHITE);
				g.fillRect(0, 0, image_.getWidth(), image_.getHeight());
			}
			g.setColor(Color.BLACK);
			for (List<int[]> line : lines_) {
				final Iterator<int[]> iter = line.iterator();
				int[] prev = iter.next();
				while (iter.hasNext()) {
					int next[] = iter.next();
					g.drawLine(prev[0], prev[1], next[0], next[1]);
					prev = next;
				}
			}
		}
	}

	public static BufferedImage matToBufferedImage(Mat matrix) {
		int cols = matrix.cols();
		int rows = matrix.rows();
		int elemSize = (int) matrix.elemSize();
		byte[] data = new byte[cols * rows * elemSize];
		int type;

		matrix.get(0, 0, data);

		switch (matrix.channels()) {
		case 1:
			type = BufferedImage.TYPE_BYTE_GRAY;
			break;

		case 3:
			type = BufferedImage.TYPE_3BYTE_BGR;

			// bgr to rgb
			byte b;
			for (int i = 0; i < data.length; i = i + 3) {
				b = data[i];
				data[i] = data[i + 2];
				data[i + 2] = b;
			}
			break;

		default:
			return null;
		}

		BufferedImage image = new BufferedImage(cols, rows, type);
		image.getRaster().setDataElements(0, 0, cols, rows, data);

		return image;
	}

}
